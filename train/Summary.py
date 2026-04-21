import os
import json
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset

# ===================== 配置 =====================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct/"
DATA_PATH = "memory.json"
OUTPUT_DIR = "output_lora_3b"

MAX_LENGTH = 512          # 可根据实际数据长度调整
BATCH_SIZE = 1
EPOCHS = 3
LR = 2e-4

# ===================== 加载数据 =====================
# 使用 json 格式读取，确保字段为 instruction, input, output
dataset = Dataset.from_json(DATA_PATH)

# 检查数据字段，若字段名不同可在此映射
# 示例数据格式：
# {
#   "instruction": "...",
#   "input": "...",
#   "output": "..."
# }
# 确保数据集中包含这些字段
print("数据样例：", dataset[0])

# ===================== 加载模型与分词器 =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Qwen2 系列建议使用 eos_token 作为 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

# ===================== LoRA 配置 =====================
# Qwen2 的线性层名称通常为 q_proj, k_proj, v_proj, o_proj 等
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],   # 可根据需要增加模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===================== 数据处理函数 =====================
def format_instruction(example):
    """将 instruction 和 input 组合成模型输入文本"""
    # 根据 Qwen2 的对话模板构造输入，此处采用简单拼接方式
    # 您也可以使用 tokenizer.apply_chat_template 来适配官方格式
    instruction = example["instruction"]
    user_input = example["input"] if example["input"] else ""
    # 构造输入文本：通常 instruction + 用户输入
    if user_input:
        text = f"<|im_start|>user\n{instruction}\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    else:
        text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    # 输出部分
    output = example["output"]
    return {"text": text, "output": output}

# 应用格式化，生成新字段
dataset = dataset.map(format_instruction)

def tokenize_function(examples):
    """对输入文本进行分词，并生成 labels（仅对 assistant 部分计算损失）"""
    # 分词输入文本（不含输出）
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,          # 暂不填充，交由 collator 统一处理
        return_tensors=None,
    )
    # 分词输出文本（加上 eos_token）
    outputs = tokenizer(
        examples["output"],
        truncation=True,
        max_length=MAX_LENGTH - len(inputs["input_ids"][0]),  # 防止总长度超限
        padding=False,
        return_tensors=None,
    )
    # 合并 input_ids 和 labels
    batch_input_ids = []
    batch_labels = []
    for i in range(len(examples["text"])):
        input_ids = inputs["input_ids"][i]
        output_ids = outputs["input_ids"][i]
        # 总长度不超过 MAX_LENGTH
        total_len = len(input_ids) + len(output_ids)
        if total_len > MAX_LENGTH:
            # 优先截断输出部分
            output_ids = output_ids[:MAX_LENGTH - len(input_ids)]
        # 合并
        full_input_ids = input_ids + output_ids
        # 标签：输入部分设为 -100（忽略），输出部分为 output_ids
        labels = [-100] * len(input_ids) + output_ids
        batch_input_ids.append(full_input_ids)
        batch_labels.append(labels)
    return {
        "input_ids": batch_input_ids,
        "labels": batch_labels,
        "attention_mask": [[1] * len(ids) for ids in batch_input_ids]  # 临时掩码，collator 会重新生成
    }

# 应用分词
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names  # 移除原始字段，保留 input_ids, labels, attention_mask
)

# ===================== 训练参数 =====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    learning_rate=LR,
    logging_steps=5,
    save_steps=100,
    fp16=False,
    bf16=False,
    # use_cpu=True,
    report_to="none",
    save_total_limit=2,          # 只保留最近两个 checkpoint
    dataloader_pin_memory=True, # CPU 训练时建议关闭
)

# 数据整理器：自动填充到批次内最大长度
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    label_pad_token_id=-100,     # 忽略标签中的填充
    padding=True,
)

# ===================== 启动训练 =====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

# 保存 LoRA 权重和分词器
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ 训练完成！模型已保存到：", OUTPUT_DIR)