import json
from typing import Any, Dict, List, Optional

from multi_agent.memory.user_summary_store import UserSummaryStore
from multi_agent.prompts.summary_prompts import INTEGRATED_SUMMARY_INSTRUCTION, L2_SESSION_SUMMARY_PROMPT
from multi_agent.utils.openai_client import OpenAIChatClient


class SummaryAgent:
    def __init__(self, openai_client: Optional[OpenAIChatClient] = None):
        self.openai_client = openai_client or OpenAIChatClient()
        self.summary_store = UserSummaryStore()

    def _call_model(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "你是一个擅长心理咨询总结与结构化信息提取的助手，输出尽量遵循用户要求。"},
            {"role": "user", "content": prompt},
        ]
        response_text, _usage = self.openai_client.chat(messages=messages)
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
