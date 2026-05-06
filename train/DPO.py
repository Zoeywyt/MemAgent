#!/usr/bin/env python3
"""
DPO训练脚本 - 适配TRL 1.0版本，基于SFT后的Qwen2.5-7B模型
数据格式: JSONL，每行 {"messages": [...], "chosen": "...", "rejected": "..."}
支持训练和独立评估两种模式：python dpo_train_eval.py          # 训练
                          python dpo_train_eval.py eval <model_path> [metrics]
"""

import importlib.util
import os
import json
import math
import sys
from collections import Counter

# Must be set before importing torch/accelerate/transformers.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging as hf_logging,
)
from trl import DPOTrainer, DPOConfig
from peft import (
    prepare_model_for_kbit_training,
    PeftModel,
)
from tqdm import tqdm
from bert_score import BERTScorer   # 确保已安装 bert-score

hf_logging.set_verbosity_info()

# ==================== 配置参数 ====================
SFT_MODEL_PATH = "output_sft_7b"
OUTPUT_DIR = "output_dpo_7b"
DATA_PATH = "cpsyCoun_dpo.jsonl"
BASE_MODEL_NAME = "/data/home/wls_cwz/data/model/Qwen/Qwen2.5-7B-Instruct/"

# LoRA配置（与SFT一致）
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# DPO参数（修改：BETA = 0.5, MAX_LENGTH = 3072）
BETA = 0.5
MAX_LENGTH = 3200
MAX_PROMPT_LENGTH = 3072              # 保持不变

# 训练配置（修改：LEARNING_RATE = 3e-6）
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 3e-6
NUM_EPOCHS = 1
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
SAVE_STEPS = 500
EVAL_STEPS = 500

# 数据配置
MAX_TRAIN_SAMPLES = None
TEST_SIZE = 0.05

# 量化配置
USE_4BIT = True
QUANT_TYPE = "nf4"
DOUBLE_QUANT = True
QUANT_COMPUTE_DTYPE = torch.bfloat16

SEED = 42
EVALUATE_AFTER_TRAIN = True
EVAL_MODEL_PATH = OUTPUT_DIR

# BERTScore 配置（修复关键点）
BERTSCORE_MODEL_TYPE = "bert-base-chinese"   # 使用中文 BERT
BERTSCORE_LANG = "zh"                         # 显式指定中文


def resolve_report_to():
    if importlib.util.find_spec("tensorboard") or importlib.util.find_spec("tensorboardX"):
        return "tensorboard"
    print("TensorBoard not installed; disabling reporting.")
    return "none"


# ==================== 数据加载 ====================
def load_preference_data(jsonl_path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if all(k in item for k in ["messages", "chosen", "rejected"]):
                    data.append({
                        "prompt": item["messages"],
                        "chosen": [{"role": "assistant", "content": item["chosen"]}],
                        "rejected": [{"role": "assistant", "content": item["rejected"]}],
                        "messages": item["messages"],
                        "chosen_text": item["chosen"],
                        "rejected_text": item["rejected"],
                    })
                else:
                    print(f"Warning: Line {line_num} missing required fields, skipped.")
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    print(f"Loaded {len(data)} preference pairs from {jsonl_path}")
    return Dataset.from_list(data)


def keep_dpo_columns(dataset):
    keep = {"prompt", "chosen", "rejected"}
    remove_columns = [col for col in dataset.column_names if col not in keep]
    return dataset.remove_columns(remove_columns) if remove_columns else dataset


# ==================== 模型加载 ====================
def setup_model_and_tokenizer():
    """加载基础模型 + SFT的LoRA适配器"""
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=QUANT_TYPE,
            bnb_4bit_use_double_quant=DOUBLE_QUANT,
            bnb_4bit_compute_dtype=QUANT_COMPUTE_DTYPE,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False,
        )

    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    # Tokenizer（优先从SFT目录加载）
    tokenizer_path = SFT_MODEL_PATH if os.path.exists(os.path.join(SFT_MODEL_PATH, "tokenizer.json")) else BASE_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载SFT的LoRA适配器，并设为可训练
    model = PeftModel.from_pretrained(model, SFT_MODEL_PATH, is_trainable=True)
    print(f"Loaded SFT adapter from {SFT_MODEL_PATH}")

    model.train()
    return model, tokenizer


def prepare_model_for_dpo(model):
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} || Total params: {total_params:,} || "
          f"Trainable%: {100 * trainable_params / total_params:.4f}")
    if trainable_params == 0:
        raise ValueError("No trainable parameters found after loading the SFT adapter.")
    return model


# ==================== 评估指标辅助函数 ====================
def _tokenize_text(text, tokenizer):
    """使用模型的tokenizer进行分词（用于BLEU/ROUGE）"""
    text = text.strip()
    if not text:
        return []
    # 简单按空格分词（英文）或按字符（中文），这里使用tokenizer的tokenize保持一致性
    return tokenizer.tokenize(text)


def compute_bleu4(predictions, references, tokenizer):
    max_order = 4
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    pred_len = 0
    ref_len = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = _tokenize_text(pred, tokenizer)
        ref_tokens = _tokenize_text(ref, tokenizer)
        pred_len += len(pred_tokens)
        ref_len += len(ref_tokens)

        for n in range(1, max_order + 1):
            pred_ngrams = Counter(tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1))
            ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
            overlap = pred_ngrams & ref_ngrams
            matches_by_order[n - 1] += sum(overlap.values())
            possible_matches_by_order[n - 1] += max(len(pred_tokens) - n + 1, 0)

    if pred_len == 0:
        return 0.0
    precisions = []
    for i in range(max_order):
        if possible_matches_by_order[i] > 0:
            precisions.append(matches_by_order[i] / possible_matches_by_order[i])
        else:
            precisions.append(1e-6)  # 避免log(0)
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / pred_len)
    return brevity_penalty * geo_mean


def compute_rouge_l(predictions, references, tokenizer):
    def lcs_len(a, b):
        if not a or not b:
            return 0
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0]
            for j, y in enumerate(b, 1):
                cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
            prev = cur
        return prev[-1]

    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = _tokenize_text(pred, tokenizer)
        ref_tokens = _tokenize_text(ref, tokenizer)
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        lcs = lcs_len(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return sum(scores) / len(scores) if scores else 0.0


def safe_exp(value, cap=1e6):
    if value is None or not math.isfinite(value):
        return cap
    if value >= math.log(cap):
        return cap
    return math.exp(value)


# ==================== 主评估函数（稳定版） ====================
def evaluate_model(model, tokenizer, eval_dataset, metrics=None):
    requested = set(metrics) if metrics else {"bleu4", "rougeL", "bertscore", "ppl", "distinct"}
    want_bleu = "bleu4" in requested
    want_rouge = "rougeL" in requested
    want_bertscore = "bertscore" in requested
    want_ppl = "ppl" in requested
    want_distinct = "distinct" in requested

    references = []
    predictions = []
    ppl_losses = []

    # 初始化 BERTScorer（修复点：显式设置 lang="zh"，model_type 使用中文 BERT）
    bert_scorer = None
    if want_bertscore:
        try:
            bert_scorer = BERTScorer(
                model_type=BERTSCORE_MODEL_TYPE,
                lang=BERTSCORE_LANG,
                rescale_with_baseline=True,
            )
            print(f"BERTScorer initialized with {BERTSCORE_MODEL_TYPE}, lang={BERTSCORE_LANG}")
        except Exception as e:
            print(f"Warning: BERTScorer init failed: {e}")

    model.eval()
    device = next(model.parameters()).device

    for sample in tqdm(eval_dataset, desc="Evaluating generation metrics"):
        messages = sample.get("messages", sample.get("prompt"))
        reference = sample.get("chosen_text")
        if reference is None:
            chosen = sample["chosen"]
            reference = chosen[0]["content"] if isinstance(chosen, list) else chosen
        if not messages or not reference:
            continue

        # 构造 prompt（使用 chat template）
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT_LENGTH,
        ).to(device)

        # 生成回复（贪婪解码，保证确定性）
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = outputs[:, inputs.input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # 计算生成部分的 Perplexity
        if gen_ids.numel() > 0:
            with torch.no_grad():
                full_input_ids = torch.cat([inputs.input_ids, gen_ids], dim=1)
                outputs2 = model(full_input_ids, labels=full_input_ids)
                shift_logits = outputs2.logits[:, :-1, :].contiguous()
                shift_labels = full_input_ids[:, 1:].contiguous()
                gen_start = inputs.input_ids.shape[1]
                gen_end = gen_start + gen_ids.shape[1]
                gen_logits = shift_logits[:, gen_start - 1:gen_end - 1, :]
                gen_labels = shift_labels[:, gen_start - 1:gen_end - 1]
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                token_losses = loss_fct(
                    gen_logits.reshape(-1, gen_logits.size(-1)),
                    gen_labels.reshape(-1),
                )
                seq_loss = token_losses.reshape(gen_labels.shape).mean().item()
                if math.isfinite(seq_loss):
                    ppl_losses.append(seq_loss)

        references.append(reference)
        predictions.append(gen_text)

    # 计算各项指标
    bleu4 = compute_bleu4(predictions, references, tokenizer) if want_bleu else None
    rouge_l = compute_rouge_l(predictions, references, tokenizer) if want_rouge else None

    bertscore_f1 = None
    if want_bertscore and bert_scorer and predictions and references:
        try:
            # 过滤空字符串（BERTScore 不能处理）
            valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
            if valid_pairs:
                preds_clean, refs_clean = zip(*valid_pairs)
                P, R, F1 = bert_scorer.score(list(preds_clean), list(refs_clean))
                bertscore_f1 = F1.mean().item()
            else:
                print("Warning: No valid non-empty predictions/references for BERTScore.")
        except Exception as e:
            print(f"BERTScore scoring failed: {e}")
            import traceback
            traceback.print_exc()

    avg_ppl = None
    if want_ppl:
        avg_loss = sum(ppl_losses) / len(ppl_losses) if ppl_losses else None
        avg_ppl = safe_exp(avg_loss)

    distinct1 = distinct2 = None
    if want_distinct:
        all_tokens_1 = []
        all_tokens_2 = []
        for pred in predictions:
            tokens = tokenizer.tokenize(pred)
            all_tokens_1.extend(tokens)
            all_tokens_2.extend(tuple(tokens[i:i+2]) for i in range(len(tokens)-1))
        distinct1 = len(set(all_tokens_1)) / len(all_tokens_1) if all_tokens_1 else 0.0
        distinct2 = len(set(all_tokens_2)) / len(all_tokens_2) if all_tokens_2 else 0.0

    # 打印结果
    print("\n" + "=" * 50)
    print("DPO Generation Evaluation Results:")
    if want_bleu:
        print(f"BLEU-4: {bleu4:.4f}")
    if want_rouge:
        print(f"ROUGE-L: {rouge_l:.4f}")
    print(f"BERTScore F1: {bertscore_f1:.4f}" if bertscore_f1 is not None else "BERTScore F1: N/A")
    if want_ppl:
        print(f"Perplexity (PPL): {avg_ppl:.2f}")
    if want_distinct:
        print(f"Distinct-1: {distinct1:.4f}")
        print(f"Distinct-2: {distinct2:.4f}")
    print("=" * 50)

    result = {"bertscore_f1": bertscore_f1}
    if want_bleu: result["bleu4"] = bleu4
    if want_rouge: result["rougeL"] = rouge_l
    if want_ppl: result["ppl"] = avg_ppl
    if want_distinct: result["distinct1"] = distinct1; result["distinct2"] = distinct2
    return result


def setup_eval_model_and_tokenizer(adapter_path):
    """加载已训练好的DPO模型用于评估"""
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=QUANT_TYPE,
            bnb_4bit_use_double_quant=DOUBLE_QUANT,
            bnb_4bit_compute_dtype=QUANT_COMPUTE_DTYPE,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False,
        )

    tokenizer_path = adapter_path if os.path.exists(os.path.join(adapter_path, "tokenizer.json")) else SFT_MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, adapter_path)
    print(f"Loaded adapter from {adapter_path}")
    return model, tokenizer


# ==================== 训练主函数 ====================
def train():
    # 1. 数据准备
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    dataset = load_preference_data(DATA_PATH)
    if len(dataset) == 0:
        raise ValueError("No valid preference pairs found.")

    if MAX_TRAIN_SAMPLES and len(dataset) > MAX_TRAIN_SAMPLES:
        print(f"Limiting dataset to {MAX_TRAIN_SAMPLES} samples")
        dataset = dataset.select(range(MAX_TRAIN_SAMPLES))

    dataset = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    eval_dataset_for_metrics = eval_dataset
    train_dataset = keep_dpo_columns(train_dataset)
    eval_dataset = keep_dpo_columns(eval_dataset)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # 2. 模型准备
    model, tokenizer = setup_model_and_tokenizer()
    model = prepare_model_for_dpo(model)

    # 3. DPO配置（注意 max_length 已改为 3072）
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        bf16=USE_4BIT,
        fp16=False,
        report_to=resolve_report_to(),
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
        disable_tqdm=False,
        log_level="info",
        # DPO 特有参数（修改：beta 和 max_length）
        beta=BETA,
        max_length=MAX_LENGTH,
    )

    # 4. 创建DPO训练器
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # 5. 训练
    print("Starting DPO training...")
    dpo_trainer.train()

    # 6. 输出训练日志
    log_history = dpo_trainer.state.log_history
    last_train_log = next((e for e in reversed(log_history) if 'train_loss' in e), None)
    last_eval_log = next((e for e in reversed(log_history) if 'eval_loss' in e), None)
    if last_train_log:
        print("\n" + str(last_train_log))
    if last_eval_log:
        print(str(last_eval_log))

    # 7. 保存模型
    dpo_trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"DPO model saved to {OUTPUT_DIR}")

    final_train_loss = last_train_log.get('train_loss', 'N/A') if last_train_log else 'N/A'
    final_eval_loss = last_eval_log.get('eval_loss', 'N/A') if last_eval_log else 'N/A'
    print(f"Final training loss: {final_train_loss}")
    print(f"Final evaluation loss: {final_eval_loss}")

    if EVALUATE_AFTER_TRAIN:
        print("\nEvaluating DPO model generation metrics on eval set...")
        evaluate_model(dpo_trainer.model, tokenizer, eval_dataset_for_metrics)

    print("DPO training completed successfully!")


def evaluate_only(model_path=EVAL_MODEL_PATH, metric_selector=None):
    """独立评估模式"""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    dataset = load_preference_data(DATA_PATH)
    if len(dataset) == 0:
        raise ValueError("No valid preference pairs found.")

    if MAX_TRAIN_SAMPLES and len(dataset) > MAX_TRAIN_SAMPLES:
        dataset = dataset.select(range(MAX_TRAIN_SAMPLES))

    dataset = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    eval_dataset = dataset["test"]
    print(f"Eval samples: {len(eval_dataset)}")

    model, tokenizer = setup_eval_model_and_tokenizer(model_path)
    if metric_selector and metric_selector != "all":
        metrics = {m.strip() for m in metric_selector.split(",") if m.strip()}
        evaluate_model(model, tokenizer, eval_dataset, metrics=metrics)
    else:
        evaluate_model(model, tokenizer, eval_dataset)


if __name__ == "__main__":
    try:
        if len(sys.argv) >= 2 and sys.argv[1] == "eval":
            eval_model_path = sys.argv[2] if len(sys.argv) >= 3 else EVAL_MODEL_PATH
            eval_metric = sys.argv[3] if len(sys.argv) >= 4 else None
            evaluate_only(eval_model_path, eval_metric)
        else:
            train()
    except Exception as e:
        print(f"Run failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)