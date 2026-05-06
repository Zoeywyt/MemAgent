#!/usr/bin/env python3
"""
SFT训练脚本 - 基于Qwen2.5-7B-Instruct，支持训练/验证/测试集划分
训练后自动评估：BLEU-4, ROUGE-L, BERTScore, Perplexity, Distinct-1/2
"""

import importlib.util
import os
import sys
import json
import math

# Must be set before importing torch/accelerate/transformers.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6,7")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    EarlyStoppingCallback,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging as hf_logging
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
try:
    from bert_score import BERTScorer
except ImportError:
    BERTScorer = None

# 设置日志
hf_logging.set_verbosity_info()

# ==================== 配置参数 ====================
MODEL_NAME = "/data/home/wls_cwz/data/model/Qwen/Qwen2.5-7B-Instruct/"
OUTPUT_DIR = "output_sft_7b"

# LoRA配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 训练配置
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
MAX_SEQ_LENGTH = 3072
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 5
SAVE_STEPS = 100
EVAL_STEPS = 100

# 数据划分比例（训练:验证:测试）
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# 量化配置
USE_4BIT = True
QUANT_TYPE = "nf4"
DOUBLE_QUANT = True
QUANT_COMPUTE_DTYPE = torch.bfloat16

# 评估开关
EVALUATE_AFTER_TRAIN = True
EVAL_MODEL_PATH = OUTPUT_DIR
BERTSCORE_MODEL_TYPE = "xlm-roberta-base"

def resolve_report_to():
    if importlib.util.find_spec("tensorboard") or importlib.util.find_spec("tensorboardX"):
        return "tensorboard"
    print("TensorBoard not installed; disabling report_to.")
    return "none"

# ==================== 数据加载与划分 ====================
def load_messages_dataset(jsonl_path):
    """加载JSONL格式的messages数据"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
                if "messages" in conv and isinstance(conv["messages"], list):
                    data.append(conv)
                else:
                    print(f"Warning: Line {line_num} missing 'messages' field")
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    print(f"Loaded {len(data)} conversations from {jsonl_path}")
    return Dataset.from_list(data)

def split_datasets(dataset):
    """按比例划分训练集、验证集、测试集"""
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-5, "比例之和必须为1"
    
    # 先分出训练+验证 与 测试
    train_val_ratio = TRAIN_RATIO + VAL_RATIO
    train_val_test = dataset.train_test_split(test_size=TEST_RATIO, seed=RANDOM_SEED)
    test_dataset = train_val_test["test"]
    train_val_dataset = train_val_test["train"]
    
    # 再从训练+验证中分出验证集
    val_ratio_adjusted = VAL_RATIO / train_val_ratio
    train_val_split = train_val_dataset.train_test_split(test_size=val_ratio_adjusted, seed=RANDOM_SEED)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    
    print(f"Split results: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset

# ==================== 格式化函数 ====================
def formatting_func(example):
    messages = example["messages"]
    if not messages or not isinstance(messages, list):
        return ""
    valid_messages = []
    for msg in messages:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            valid_messages.append(msg)
    if not valid_messages:
        return ""
    try:
        text = tokenizer.apply_chat_template(
            valid_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return text
    except Exception:
        # 回退手动格式化
        parts = []
        for msg in valid_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        return "\n".join(parts)

# ==================== 模型加载 ====================
def setup_model_and_tokenizer():
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=QUANT_TYPE,
            bnb_4bit_use_double_quant=DOUBLE_QUANT,
            bnb_4bit_compute_dtype=QUANT_COMPUTE_DTYPE,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False
        )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
        truncation_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def setup_lora_model(model):
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# ==================== 评估函数 ====================
def _metric_tokens(text):
    """Tokenize for lightweight local text metrics."""
    text = text.strip()
    if not text:
        return []
    if any(ch.isspace() for ch in text):
        return text.split()
    return list(text)

def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

def compute_bleu4(predictions, references):
    """Corpus BLEU-4 with add-one smoothing, avoiding evaluate.load('bleu')."""
    max_order = 4
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    pred_len = 0
    ref_len = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = _metric_tokens(pred)
        ref_tokens = _metric_tokens(ref)
        pred_len += len(pred_tokens)
        ref_len += len(ref_tokens)

        for n in range(1, max_order + 1):
            pred_counts = _ngram_counts(pred_tokens, n)
            ref_counts = _ngram_counts(ref_tokens, n)
            overlap = pred_counts & ref_counts
            matches_by_order[n - 1] += sum(overlap.values())
            possible_matches_by_order[n - 1] += max(len(pred_tokens) - n + 1, 0)

    if pred_len == 0:
        return 0.0

    precisions = [
        (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
        for i in range(max_order)
    ]
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / pred_len)
    return brevity_penalty * geo_mean

def _lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]

def compute_rouge_l(predictions, references):
    """Mean ROUGE-L F1."""
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = _metric_tokens(pred)
        ref_tokens = _metric_tokens(ref)
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        lcs = _lcs_len(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0

def safe_exp(value, cap=1e6):
    if value is None or not math.isfinite(value):
        return cap
    if value >= math.log(cap):
        return cap
    return math.exp(value)

def evaluate_model(model, tokenizer, test_dataset):
    """
    在全部测试集上计算：
    - BLEU-4
    - ROUGE-L
    - BERTScore F1 (中文)
    - Perplexity (生成部分)
    - Distinct-1 / Distinct-2 (全局多样性)
    """
    references = []
    predictions = []
    ppl_losses = []      # 每个生成序列的平均交叉熵损失
    empty_generations = 0
    bertscore_error = None

    # 中文建议使用 lang="zh"，若数据为英文可改回 "en"。
    # BERTScore needs local model/cache availability, so keep it best-effort.
    bert_scorer = None
    if BERTScorer is None:
        bertscore_error = "bert_score is not installed"
        print(f"Warning: BERTScore disabled: {bertscore_error}.")
    else:
        try:
            bert_scorer = BERTScorer(
                model_type=BERTSCORE_MODEL_TYPE,
                lang=None,
                rescale_with_baseline=False,
            )
        except Exception as exc:
            bertscore_error = str(exc)
            print(f"Warning: BERTScore disabled: {bertscore_error}")
    
    model.eval()
    device = next(model.parameters()).device
    
    for sample in tqdm(test_dataset, desc="Evaluating on test set"):
        messages = sample["messages"]
        if not messages:
            continue
        
        # 提取最后一个 assistant 消息作为参考
        last_assistant = None
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                last_assistant = msg["content"]
                break
        if last_assistant is None:
            continue
        
        # 构造上下文（去掉最后一个 assistant）
        context = messages[:-1] if messages[-1]["role"] == "assistant" else messages
        prompt = tokenizer.apply_chat_template(
            context,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(device)
        
        # 生成回复（贪婪解码）
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        gen_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        
        # 计算生成部分的 Perplexity
        if gen_ids.numel() > 0:
            with torch.no_grad():
                full_input_ids = torch.cat([inputs.input_ids, gen_ids], dim=1)
                outputs2 = model(full_input_ids, labels=full_input_ids)
                logits = outputs2.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = full_input_ids[:, 1:].contiguous()
                # 只计算生成部分的 loss
                gen_start = inputs.input_ids.shape[1]
                gen_end = gen_start + gen_ids.shape[1]
                # shift后，生成部分的标签对应位置: [gen_start-1, gen_end-1)
                gen_logits = shift_logits[:, gen_start-1:gen_end-1, :]
                gen_labels = shift_labels[:, gen_start-1:gen_end-1]
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(gen_logits.reshape(-1, gen_logits.size(-1)), gen_labels.reshape(-1))
                token_losses = token_losses.reshape(gen_labels.shape)
                seq_loss = token_losses.mean().item()
                if math.isfinite(seq_loss):
                    ppl_losses.append(seq_loss)
        else:
            empty_generations += 1
        
        references.append(last_assistant)
        predictions.append(gen_text)
    
    # BLEU-4 / ROUGE-L
    bleu4 = compute_bleu4(predictions, references)
    rouge_l = compute_rouge_l(predictions, references)
    # BERTScore
    if bert_scorer and predictions:
        try:
            P, R, F1 = bert_scorer.score(predictions, references)
            bertscore_f1 = F1.mean().item()
        except Exception as exc:
            bertscore_error = str(exc)
            print(f"Warning: BERTScore scoring failed: {bertscore_error}")
            bertscore_f1 = None
    else:
        bertscore_f1 = None
    # Perplexity
    avg_loss = sum(ppl_losses) / len(ppl_losses) if ppl_losses else None
    avg_ppl = safe_exp(avg_loss)
    if empty_generations:
        print(f"Warning: skipped PPL for {empty_generations} empty generations.")
    # Distinct-1/2 (全局)
    all_tokens_1 = []
    all_tokens_2 = []
    for pred in predictions:
        tokens = tokenizer.tokenize(pred)
        all_tokens_1.extend(tokens)
        bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
        all_tokens_2.extend(bigrams)
    distinct1 = len(set(all_tokens_1)) / len(all_tokens_1) if all_tokens_1 else 0.0
    distinct2 = len(set(all_tokens_2)) / len(all_tokens_2) if all_tokens_2 else 0.0
    
    print("\n" + "="*50)
    print("Test Set Evaluation Results:")
    print(f"BLEU-4: {bleu4:.4f}")
    print(f"ROUGE-L: {rouge_l:.4f}")
    print(f"BERTScore F1: {bertscore_f1:.4f}" if bertscore_f1 is not None else "BERTScore F1: N/A")
    if bertscore_error:
        print(f"BERTScore error: {bertscore_error}")
    print(f"Perplexity (PPL): {avg_ppl:.2f}")
    print(f"Distinct-1: {distinct1:.4f}")
    print(f"Distinct-2: {distinct2:.4f}")
    print("="*50)
    
    return {
        "bleu4": bleu4,
        "rougeL": rouge_l,
        "bertscore_f1": bertscore_f1,
        "bertscore_error": bertscore_error,
        "ppl": avg_ppl,
        "distinct1": distinct1,
        "distinct2": distinct2
    }

# ==================== 训练主函数 ====================
def train():
    global tokenizer
    
    data_path = "sft.jsonl"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # 加载并划分数据集
    full_dataset = load_messages_dataset(data_path)
    if len(full_dataset) == 0:
        raise ValueError("No valid conversations found")
    train_dataset, val_dataset, test_dataset = split_datasets(full_dataset)
    
    # 初始化模型
    model, tokenizer = setup_model_and_tokenizer()
    model = setup_lora_model(model)
    
    # 训练参数
    training_args = SFTConfig(
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
        bf16=False,
        fp16=False,
        report_to=resolve_report_to(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        disable_tqdm=False,
        logging_first_step=True,
        log_level="info",
    )
    
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        callbacks=[early_stopping],
    )
    
    print("Starting training...")
    # trainer.train()
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    trainer.train(resume_from_checkpoint=last_checkpoint)


    
    # 输出训练过程最后记录
    log_history = trainer.state.log_history
    last_train = next((e for e in reversed(log_history) if 'train_loss' in e), None)
    last_eval = next((e for e in reversed(log_history) if 'eval_loss' in e), None)
    if last_train:
        print("\n" + str(last_train))
    if last_eval:
        print(str(last_eval))
    
    # 保存模型
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    
    if last_train:
        print(f"Final training loss: {last_train.get('train_loss', 'N/A')}")
    
    # 测试集评估
    if EVALUATE_AFTER_TRAIN:
        print("\nEvaluating on full test set...")
        evaluate_model(trainer.model, tokenizer, test_dataset)
    
    print("Training completed successfully!")

def evaluate_only(model_path=EVAL_MODEL_PATH, split="test"):
    global tokenizer

    data_path = "sft.jsonl"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")

    full_dataset = load_messages_dataset(data_path)
    if len(full_dataset) == 0:
        raise ValueError("No valid conversations found")
    _, val_dataset, test_dataset = split_datasets(full_dataset)
    eval_dataset = val_dataset if split == "val" else test_dataset

    model, tokenizer = setup_model_and_tokenizer()
    model = PeftModel.from_pretrained(model, model_path)
    print(f"Loaded adapter from {model_path}")
    print(f"Evaluating on {split} set: {len(eval_dataset)} samples")
    evaluate_model(model, tokenizer, eval_dataset)

if __name__ == "__main__":
    try:
        if len(sys.argv) >= 2 and sys.argv[1] == "eval":
            eval_model_path = sys.argv[2] if len(sys.argv) >= 3 else EVAL_MODEL_PATH
            eval_split = sys.argv[3] if len(sys.argv) >= 4 else "test"
            evaluate_only(eval_model_path, eval_split)
        else:
            train()
    except Exception as e:
        print(f"Run failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
