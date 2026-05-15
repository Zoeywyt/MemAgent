import importlib.util
import json
import os
from typing import Any, Dict, List, Optional

from memory.user_summary_store import UserSummaryStore
from prompts.summary_prompts import INTEGRATED_SUMMARY_INSTRUCTION, L2_SESSION_SUMMARY_PROMPT
from utils.model_client import ChatClientProtocol, build_chat_client
from utils.model_runtime import call_model


class SummaryAgent:
    def __init__(
        self,
        openai_client: Optional[ChatClientProtocol] = None,
        model_backend: Optional[str] = None,
        model_mode: Optional[str] = None,
        local_model_path: Optional[str] = None,
        local_base_model_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.model_backend = model_backend
        self.model_mode = model_mode
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._fallback_client: Optional[ChatClientProtocol] = None
        self.openai_client = openai_client or build_chat_client(
            "SUMMARY_AGENT",
            backend=model_backend,
            mode=model_mode,
            local_model_path=local_model_path,
            local_base_model_path=local_base_model_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_new_tokens=512,
        )
        self.summary_store = UserSummaryStore()

    def _is_local_summary_backend(self) -> bool:
        backend = str(self.model_backend or "").strip().lower().replace("-", "").replace("_", "")
        mode = str(self.model_mode or "").strip().lower()
        return backend in {"qwen3b", "qwen7b", "local"} or mode in {"local", "hf", "huggingface"}

    def _local_summary_allowed(self) -> bool:
        if not self._is_local_summary_backend():
            return True
        if os.getenv("MEMAGENT_ALLOW_CPU_SUMMARY_LOCAL", "0").strip().lower() in {"1", "true", "yes", "on"}:
            return True
        if importlib.util.find_spec("torch") is None:
            return False
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _get_fallback_client(self) -> ChatClientProtocol:
        if self._fallback_client is None:
            self._fallback_client = build_chat_client(
                "SUMMARY_AGENT_FALLBACK",
                backend="gpt",
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                max_new_tokens=512,
            )
        return self._fallback_client

    def _call_model(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "你是一个擅长心理咨询总结与结构化信息提取的助手，输出尽量遵循用户要求。"},
            {"role": "user", "content": prompt},
        ]
        if not self._local_summary_allowed():
            print(
                "[MemAgent Summary] local summary model is skipped because CUDA is unavailable; fallback to unified API.",
                flush=True,
            )
            response_text, _usage = call_model(
                self._get_fallback_client(),
                component="summary_fallback",
                messages=messages,
            )
            return response_text
        try:
            response_text, _usage = call_model(
                self.openai_client,
                component="summary",
                messages=messages,
            )
        except BaseException as exc:
            if not self._is_local_summary_backend():
                raise
            print(
                f"[MemAgent Summary] local summary model failed ({type(exc).__name__}: {exc}); fallback to unified API.",
                flush=True,
            )
            response_text, _usage = call_model(
                self._get_fallback_client(),
                component="summary_fallback",
                messages=messages,
            )
        return response_text

    def _parse_json(self, response: str) -> Optional[Dict[str, Any]]:
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except Exception:
            return None
        return None

    def generate_l2_summary(self, session_messages: List[Dict[str, str]]) -> Dict[str, Any]:
        lines = []
        turn = 1
        for i in range(0, len(session_messages), 2):
            user_msg = session_messages[i] if i < len(session_messages) else None
            assistant_msg = session_messages[i + 1] if i + 1 < len(session_messages) else None
            if user_msg:
                lines.append(f"轮次{turn} 来访者：{user_msg['content']}")
            if assistant_msg:
                lines.append(f"轮次{turn} 咨询师：{assistant_msg['content']}")
            turn += 1
        conv_text = "\n".join(lines)
        prompt = L2_SESSION_SUMMARY_PROMPT.format(conversation_history=conv_text)
        response = self._call_model(prompt)
        parsed = self._parse_json(response)
        if parsed and "topic" in parsed:
            parsed.setdefault("background", "")
            parsed.setdefault("summary", "")
            return parsed
        return {
            "topic": "未知",
            "background": f"包含{len(session_messages) // 2}轮对话",
            "summary": response[:300],
        }

    def update_l1_summary(self, user_id: str, l2_summary: Dict[str, Any]) -> Dict[str, Any]:
        existing = self.summary_store.get_user_summary(user_id)
        if existing is None:
            new_l1 = {
                "主题": l2_summary.get("topic", "未知"),
                "背景": l2_summary.get("background", ""),
                "会话总结": l2_summary.get("summary", ""),
            }
        else:
            prompt = INTEGRATED_SUMMARY_INSTRUCTION.format(
                existing_summary=json.dumps(existing, ensure_ascii=False, indent=2),
                new_session_summary=json.dumps(
                    {
                        "主题": l2_summary.get("topic"),
                        "背景": l2_summary.get("background"),
                        "会话总结": l2_summary.get("summary"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            response = self._call_model(prompt)
            parsed = self._parse_json(response)
            if parsed and "主题" in parsed:
                new_l1 = parsed
            else:
                new_l1 = existing
        self.summary_store.save_user_summary(user_id, new_l1)
        return new_l1

    def get_user_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self.summary_store.get_user_summary(user_id)
