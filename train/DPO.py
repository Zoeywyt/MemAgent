#!/usr/bin/env python3
"""
DPO训练脚本 - 适配TRL 1.0版本，基于SFT后的Qwen2.5-7B模型
数据格式: JSONL，每行 {"messages": [...], "chosen": "...", "rejected": "..."}
"""

import importlib.util
import os
import torch
import json
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

hf_logging.set_verbosity_info()

# ==================== 配置参数 ====================
SFT_MODEL_PATH = "output_sft_7b"
OUTPUT_DIR = "output_dpo_7b"
DATA_PATH = "preference_30k.jsonl"
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct/"

# LoRA配置（与SFT一致）
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# DPO参数
BETA = 0.1
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512

# 训练配置
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 5e-5
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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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
                        "messages": item["messages"],
                        "chosen": item["chosen"],
                        "rejected": item["rejected"]
                    })
                else:
                    print(f"Warning: Line {line_num} missing required fields, skipped.")
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    print(f"Loaded {len(data)} preference pairs from {jsonl_path}")
    return Dataset.from_list(data)


# ==================== 模型加载 ====================
def setup_model_and_tokenizer():
    """加载基础模型 + SFT的LoRA适配器"""
    # 1. 基础模型（量化）
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

    # 2. Tokenizer（优先从SFT目录加载，缺失则回退基础模型）
    tokenizer_path = SFT_MODEL_PATH if os.path.exists(os.path.join(SFT_MODEL_PATH, "tokenizer.json")) else BASE_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 加载SFT的LoRA适配器
    model = PeftModel.from_pretrained(model, SFT_MODEL_PATH)
    print(f"Loaded SFT adapter from {SFT_MODEL_PATH}")

    # 4. 确保适配器处于可训练状态
    model.train()
    for module in model.modules():
        if hasattr(module, "enable_adapters"):
            module.enable_adapters(True)
    return model, tokenizer


def prepare_model_for_dpo(model):
    """4bit训练准备 & 梯度检查点"""
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} || Total params: {total_params:,} || "
          f"Trainable%: {100 * trainable_params / total_params:.4f}")
    return model


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
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # 2. 模型准备
    model, tokenizer = setup_model_and_tokenizer()
    model = prepare_model_for_dpo(model)

    # 3. DPO配置（TRL 1.0 所有参数均在DPOConfig中）
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
        # DPO 特有参数（TRL 1.0 放在此处）
        beta=BETA,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
    )

    # 4. 创建DPO训练器（TRL 1.0 使用 processing_class）
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,                     # 自动复制模型并冻结
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,         # TRL 1.0 正确参数名
    )

    # 5. 训练
    print("Starting DPO training...")
    print(f"Beta = {BETA} | LR = {LEARNING_RATE} | Epochs = {NUM_EPOCHS}")
    dpo_trainer.train()

    # 6. 提取并打印最终损失
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

    print("DPO training completed successfully!")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)