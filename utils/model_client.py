from __future__ import annotations

import json
import os
import threading
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Protocol, Tuple

import torch
from dotenv import load_dotenv

from utils.openai_client import OpenAIChatClient


ChatMessages = List[Dict[str, str]]


class ChatClientProtocol(Protocol):
    def chat(
        self,
        messages: ChatMessages,
        print_stream: bool = False,
    ) -> Tuple[str, Optional[Dict[str, object]]]:
        ...

    def stream_chat(self, messages: ChatMessages) -> Iterator[str]:
        ...


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_local_model_path() -> str:
    return str(_project_root() / "models" / "Qwen2.5-3B-Instruct-Lora")


def _models_root() -> Path:
    return _project_root() / "models"


LOCAL_MODEL_PRESETS: Dict[str, Dict[str, str]] = {
    "qwen3b": {
        "model_path": str(_models_root() / "Qwen2.5-3B-Instruct-Lora"),
        "base_model_path": str(_models_root() / "Qwen2.5-3B-Instruct"),
    },
    "qwen7b": {
        "model_path": str(_models_root() / "Qwen2.5-7B-Instruct-DPO"),
        "base_model_path": str(_models_root() / "Qwen2.5-7B-Instruct"),
    },
}


def _normalize_mode(mode: Optional[str], default: str = "remote") -> str:
    value = (mode or default).strip().lower()
    aliases = {
        "api": "remote",
        "gpt": "remote",
        "openai": "remote",
        "remote": "remote",
        "hf": "local",
        "huggingface": "local",
        "local": "local",
    }
    return aliases.get(value, default)


def _normalize_backend(backend: Optional[str], default: str = "gpt") -> str:
    value = (backend or default).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "api": "gpt",
        "gpt": "gpt",
        "openai": "gpt",
        "remote": "gpt",
        "local": "local",
        "customlocal": "local",
        "qwen3b": "qwen3b",
        "3b": "qwen3b",
        "qwen25": "qwen3b",
        "qwen25o3b": "qwen3b",
        "qwen25-3b": "qwen3b",
        "qwen25-3binstruct": "qwen3b",
        "qwen7b": "qwen7b",
        "7b": "qwen7b",
        "qwen25o7b": "qwen7b",
        "qwen25-7b": "qwen7b",
        "qwen25-7binstruct": "qwen7b",
    }
    return aliases.get(value, default)


def resolve_model_backend(component_name: str, explicit_backend: Optional[str] = None, default: str = "gpt") -> str:
    load_dotenv()
    env_backend = os.getenv(f"{component_name}_MODEL_BACKEND") or os.getenv("MULTI_AGENT_MODEL_BACKEND")
    return _normalize_backend(explicit_backend or env_backend, default=default)


def resolve_local_preset(component_name: str, backend: str) -> Dict[str, Optional[str]]:
    preset = LOCAL_MODEL_PRESETS.get(backend, {})
    model_path = os.getenv(f"{component_name}_LOCAL_MODEL_PATH") or os.getenv("LOCAL_MODEL_PATH") or preset.get("model_path")
    base_model_path = (
        os.getenv(f"{component_name}_LOCAL_BASE_MODEL_PATH")
        or os.getenv("LOCAL_BASE_MODEL_PATH")
        or preset.get("base_model_path")
    )
    return {
        "model_path": str(Path(model_path).expanduser()) if model_path else None,
        "base_model_path": str(Path(base_model_path).expanduser()) if base_model_path else None,
    }


def resolve_model_mode(component_name: str, explicit_mode: Optional[str] = None, default: str = "remote") -> str:
    load_dotenv()
    env_mode = os.getenv(f"{component_name}_MODEL_MODE") or os.getenv("MULTI_AGENT_MODEL_MODE")
    return _normalize_mode(explicit_mode or env_mode, default=default)


def resolve_local_model_path(component_name: str, explicit_path: Optional[str] = None) -> str:
    load_dotenv()
    env_path = os.getenv(f"{component_name}_LOCAL_MODEL_PATH") or os.getenv("LOCAL_MODEL_PATH")
    candidate = explicit_path or env_path or default_local_model_path()
    return str(Path(candidate).expanduser())


def resolve_local_base_model_path(component_name: str, explicit_path: Optional[str] = None) -> Optional[str]:
    load_dotenv()
    env_path = os.getenv(f"{component_name}_LOCAL_BASE_MODEL_PATH") or os.getenv("LOCAL_BASE_MODEL_PATH")
    candidate = explicit_path or env_path
    if not candidate:
        return None
    candidate_path = Path(candidate).expanduser()
    if candidate_path.exists():
        return str(candidate_path.resolve())
    return str(candidate)


def _normalize_model_reference(reference: Optional[str]) -> str:
    if not reference:
        return ""
    candidate = Path(reference).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return str(reference)


class _SharedLocalModel:
    _instances: Dict[str, "_SharedLocalModel"] = {}
    _instances_lock = threading.Lock()

    def __new__(cls, model_path: str, base_model_path: Optional[str] = None) -> "_SharedLocalModel":
        normalized = f"{_normalize_model_reference(model_path)}::{_normalize_model_reference(base_model_path)}"
        with cls._instances_lock:
            instance = cls._instances.get(normalized)
            if instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[normalized] = instance
        return instance

    def __init__(self, model_path: str, base_model_path: Optional[str] = None) -> None:
        if getattr(self, "_initialized", False):
            return

        self.model_path = str(Path(model_path).expanduser().resolve())
        self.base_model_path = _normalize_model_reference(base_model_path)
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._initialized = True
        self._loaded = False
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return

            model_dir = Path(self.model_path)
            if not model_dir.exists():
                raise FileNotFoundError(f"Local model path does not exist: {self.model_path}")

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for local model inference. Please install it first."
                ) from exc

            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )

            adapter_config = model_dir / "adapter_config.json"
            load_error: Optional[Exception] = None

            if adapter_config.exists():
                adapter_base_model = self._read_adapter_base_model(adapter_config)
                preferred_base_model = self.base_model_path or adapter_base_model
                if importlib.util.find_spec("peft") is None:
                    if preferred_base_model:
                        self.model = self._load_plain_model(preferred_base_model, dtype)
                else:
                    try:
                        from peft import AutoPeftModelForCausalLM

                        self.model = AutoPeftModelForCausalLM.from_pretrained(
                            self.model_path,
                            dtype=dtype,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                        )
                    except Exception as exc:
                        load_error = exc
                        if preferred_base_model:
                            self.model = self._load_peft_with_base_model(preferred_base_model, dtype)

            if self.model is None:
                if load_error is not None:
                    raise RuntimeError(
                        "Failed to load the local model as both a PEFT adapter and a standard causal LM. "
                        "If this directory is a LoRA adapter, please also set LOCAL_BASE_MODEL_PATH "
                        "or <COMPONENT>_LOCAL_BASE_MODEL_PATH to the base model directory."
                    ) from load_error
                try:
                    self.model = self._load_plain_model(self.model_path, dtype)
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to load the local model. Please make sure the model directory is complete."
                    ) from exc

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model.to(self.device)
            self.model.eval()
            self._loaded = True

    def _read_adapter_base_model(self, adapter_config_path: Path) -> Optional[str]:
        try:
            data = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        base_model = data.get("base_model_name_or_path")
        return _normalize_model_reference(base_model)

    def _load_peft_with_base_model(self, base_model_path: str, dtype: torch.dtype):
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "Both peft and transformers are required to load the configured LoRA local model."
            ) from exc

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        return PeftModel.from_pretrained(
            base_model,
            self.model_path,
            dtype=dtype,
            trust_remote_code=True,
        )

    def _load_plain_model(self, model_path: str, dtype: torch.dtype):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    def generate(
        self,
        messages: ChatMessages,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[str, Dict[str, int]]:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with self._generate_lock:
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **generation_kwargs)

        input_length = int(inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0][input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        usage = {
            "prompt_tokens": input_length,
            "completion_tokens": int(generated_ids.shape[-1]),
            "total_tokens": int(output_ids.shape[-1]),
        }
        return text, usage

    def stream_generate(
        self,
        messages: ChatMessages,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Iterator[str]:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        try:
            from transformers import TextIteratorStreamer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for local model streaming inference. Please install it first."
            ) from exc

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
            "streamer": TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            ),
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        streamer = generation_kwargs["streamer"]
        error_holder: List[BaseException] = []

        def _run_generation() -> None:
            try:
                with self._generate_lock:
                    with torch.no_grad():
                        self.model.generate(**inputs, **generation_kwargs)
            except BaseException as exc:
                error_holder.append(exc)

        thread = threading.Thread(target=_run_generation, daemon=True)
        thread.start()
        for piece in streamer:
            if piece:
                yield piece
        thread.join()
        if error_holder:
            raise error_holder[0]


@dataclass
class LocalChatClient:
    model_path: str = default_local_model_path()
    base_model_path: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 1024

    def __post_init__(self) -> None:
        load_dotenv()
        self.runtime = _SharedLocalModel(self.model_path, self.base_model_path)

    def chat(
        self,
        messages: ChatMessages,
        print_stream: bool = False,
    ) -> Tuple[str, Optional[Dict[str, object]]]:
        text, usage = self.runtime.generate(
            messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        if print_stream and text:
            print(text)
        return text, usage

    def stream_chat(self, messages: ChatMessages) -> Iterator[str]:
        yield from self.runtime.stream_generate(
            messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def warmup(self) -> None:
        list(
            self.runtime.stream_generate(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "ping"},
                ],
                max_new_tokens=1,
                temperature=0.0,
                top_p=1.0,
            )
        )


def build_chat_client(
    component_name: str,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    local_model_path: Optional[str] = None,
    local_base_model_path: Optional[str] = None,
    default_mode: str = "remote",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> ChatClientProtocol:
    resolved_backend = resolve_model_backend(component_name, explicit_backend=backend, default="gpt")
    if resolved_backend in LOCAL_MODEL_PRESETS:
        preset = resolve_local_preset(component_name, resolved_backend)
        return LocalChatClient(
            model_path=local_model_path or preset["model_path"] or default_local_model_path(),
            base_model_path=local_base_model_path or preset["base_model_path"],
            max_new_tokens=max_new_tokens or 1024,
        )

    resolved_mode = resolve_model_mode(component_name, explicit_mode=mode, default=default_mode)
    if resolved_backend == "local" or resolved_mode == "local":
        return LocalChatClient(
            model_path=resolve_local_model_path(component_name, explicit_path=local_model_path),
            base_model_path=resolve_local_base_model_path(component_name, explicit_path=local_base_model_path),
            max_new_tokens=max_new_tokens or 1024,
        )

    return OpenAIChatClient(
        base_url=base_url or os.getenv(f"{component_name}_BASE_URL", "") or os.getenv("OPENAI_BASE_URL", ""),
        api_key=api_key or os.getenv(f"{component_name}_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""),
        model=model or os.getenv(f"{component_name}_MODEL", "") or os.getenv("OPENAI_MODEL", "gpt-5.4"),
        max_tokens=max_new_tokens or 1024,
    )
