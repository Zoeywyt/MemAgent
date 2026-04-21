#!/usr/bin/env python3
"""
SFT训练脚本 - 适配7B模型 + messages格式数据（24GB显存优化版 + 改进超参数）
数据格式: {"messages": [{"role": "system/user/assistant", "content": "..."}]}
改进点：
- LoRA rank 16 (原8)
- 扩展 LoRA 目标模块（含 FFN 层）
- 学习率 1e-4 (原3e-5)
- 训练轮数 5 (原3)
- 梯度裁剪 1.0
- Warmup ratio 0.1 (原0.03)
"""

import importlib.util
import os
import sys
import logging

# 设置 transformers 日志级别为 INFO，确保打印完整日志
from transformers import logging as hf_logging
hf_logging.set_verbosity_info()

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import json
from datasets import Dataset
from transformers import EarlyStoppingCallback, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ==================== 配置参数（改进版）====================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct/"
OUTPUT_DIR = "output_sft_7b"

# LoRA配置（rank=16，覆盖更多模块）
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 训练配置
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8                    # 有效batch size = 8
LEARNING_RATE = 1e-4                         # 调整学习率
NUM_EPOCHS = 5                               # 增加训练轮数
MAX_SEQ_LENGTH = 1024
WARMUP_RATIO = 0.1                           # 提升 warmup 比例
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 5
SAVE_STEPS = 100
EVAL_STEPS = 50

# 数据配置
MAX_TRAIN_SAMPLES = None
TEST_SIZE = 0.05

# 量化配置
USE_4BIT = True
QUANT_TYPE = "nf4"
DOUBLE_QUANT = True
QUANT_COMPUTE_DTYPE = torch.bfloat16


def resolve_report_to():
    if importlib.util.find_spec("tensorboard") or importlib.util.find_spec("tensorboardX"):
        return "tensorboard"
    print("TensorBoard is not installed; disabling Trainer report_to integration.")
    return "none"

# ==================== 数据加载 ====================
def load_messages_dataset(jsonl_path):
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
                    print(f"Warning: Line {line_num} missing 'messages' field or invalid format")
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    print(f"Loaded {len(data)} conversations from {jsonl_path}")
    return Dataset.from_list(data)

def formatting_func(example):
    messages = example["messages"]
    if not messages or not isinstance(messages, list):
        return ""
    valid_messages = []
    for msg in messages:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            valid_messages.append(msg)
        else:
            print(f"Warning: Invalid message format: {msg}")
    if not valid_messages:
        return ""
    try:
        text = tokenizer.apply_chat_template(
            valid_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return text
    except Exception as e:
        print(f"Error applying chat template: {e}")
        formatted_parts = []
        for msg in valid_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        return "\n".join(formatted_parts)

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

# ==================== 训练 ====================
def train():
    global tokenizer
    data_path = "sft.jsonl"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    dataset = load_messages_dataset(data_path)
    if len(dataset) == 0:
        raise ValueError("No valid conversations found in dataset")
    
    if MAX_TRAIN_SAMPLES and len(dataset) > MAX_TRAIN_SAMPLES:
        print(f"Limiting dataset to {MAX_TRAIN_SAMPLES} samples")
        dataset = dataset.select(range(MAX_TRAIN_SAMPLES))
    
    dataset = dataset.train_test_split(test_size=TEST_SIZE, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    model, tokenizer = setup_model_and_tokenizer()
    model = setup_lora_model(model)
    
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
        max_grad_norm=1.0,                     # 梯度裁剪
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
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        callbacks=[early_stopping],
    )
    
    print("Starting training...")
    trainer.train()
    
    # 训练完成后，从历史日志中提取最终的 train 和 eval 完整记录并打印
    log_history = trainer.state.log_history
    last_train_log = None
    for entry in reversed(log_history):
        if 'train_loss' in entry:
            last_train_log = entry
            break
    last_eval_log = None
    for entry in reversed(log_history):
        if 'eval_loss' in entry:
            last_eval_log = entry
            break
    
    if last_train_log:
        print("\n" + str(last_train_log))
    if last_eval_log:
        print(str(last_eval_log))
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    
    if last_train_log:
        final_train_loss = last_train_log.get('train_loss', 'N/A')
        print(f"Final training loss: {final_train_loss}")
    else:
        final_loss = trainer.state.log_history[-1].get('train_loss', 'N/A') if trainer.state.log_history else 'N/A'
        print(f"Final training loss: {final_loss}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)