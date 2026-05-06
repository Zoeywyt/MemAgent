from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    name: str
    adapter_path: Path


class LocalLoRARunner:
    def __init__(
        self,
        base_model: Path,
        adapter_path: Path,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        load_in_4bit: bool = False,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.load_in_4bit = load_in_4bit
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        if not self.base_model.exists():
            raise FileNotFoundError(f"Base model directory not found: {self.base_model}")
        if not self.adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter directory not found: {self.adapter_path}")
        if importlib.util.find_spec("peft") is None:
            raise ImportError("peft is required to compare SFT/DPO LoRA adapters.")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer_path = self.adapter_path if (self.adapter_path / "tokenizer.json").exists() else self.base_model
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True, "local_files_only": True}
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        base = AutoModelForCausalLM.from_pretrained(str(self.base_model), **kwargs)
        self.model = PeftModel.from_pretrained(base, str(self.adapter_path), local_files_only=True).eval()

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None
        import torch

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class LocalBaseRunner:
    def __init__(
        self,
        base_model: Path,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        load_in_4bit: bool = False,
    ) -> None:
        self.base_model = base_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.load_in_4bit = load_in_4bit
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        if not self.base_model.exists():
            raise FileNotFoundError(f"Base model directory not found: {self.base_model}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.base_model), trust_remote_code=True, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True, "local_files_only": True}
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(str(self.base_model), **kwargs).eval()

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None
        import torch

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class DryRunRunner:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, messages: list[dict[str, str]]) -> str:
        user_text = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if self.model_name == "dpo":
            return (
                f"听起来这件事对你影响很深。你提到“{user_text[:40]}”，"
                "我们可以先确认你的感受，再把压力拆成一两个可处理的行动。"
            )
        return (
            f"我理解你现在并不轻松。你提到“{user_text[:40]}”，"
            "这很值得被认真对待，我会陪你一步一步梳理。"
        )
