import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from prompts.supervisor_prompts import TREATMENT_PROGRESS_PROMPT
from utils.model_client import ChatClientProtocol, build_chat_client
from utils.model_runtime import call_model

if TYPE_CHECKING:
    from memory.mem0_adapter import Mem0Adapter


class SupervisorAgent:
    def __init__(
        self,
        user_id: str,
        mem0: Optional["Mem0Adapter"] = None,
        openai_client: Optional[ChatClientProtocol] = None,
        model_backend: Optional[str] = None,
        model_mode: Optional[str] = None,
        local_model_path: Optional[str] = None,
        local_base_model_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.user_id = user_id
        self._mem0 = mem0
        self.openai_client = openai_client or build_chat_client(
            "SUPERVISOR_AGENT",
            backend=model_backend,
            mode=model_mode,
            local_model_path=local_model_path,
            local_base_model_path=local_base_model_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_new_tokens=512,
        )

    @property
    def mem0(self) -> "Mem0Adapter":
        if self._mem0 is None:
            from memory.mem0_adapter import Mem0Adapter

            self._mem0 = Mem0Adapter()
        return self._mem0

    def _call_model(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "你是一位心理咨询督导师，请根据提供的长期总结与会话摘要生成结构化治疗进展报告。"},
            {"role": "user", "content": prompt},
        ]
        response_text, _usage = call_model(
            self.openai_client,
            component="supervisor",
            messages=messages,
        )
        return response_text

    def retrieve_context_for_response(self, user_input: str) -> Dict[str, Any]:
        retrieval = self.mem0.search_relevant_context(self.user_id, user_input, limit=6)
        context_text = retrieval.get("context_text", "")

        if not context_text:
            sections: List[str] = []

            if retrieval.get("l3_fragments"):
                sections.append("【相关对话片段】")
                for l3 in retrieval["l3_fragments"]:
                    timestamp = (l3.get("timestamp") or "")[:10]
                    if timestamp:
                        sections.append(f"[{timestamp}] {l3.get('memory', '')}")
                    else:
                        sections.append(f"{l3.get('memory', '')}")

            if retrieval.get("graph_relations"):
                sections.append("【相关图关系】")
                sections.extend(
                    f"- {item.get('source', '')} --{item.get('relationship', '')}--> {item.get('target', '')}"
                    for item in retrieval["graph_relations"]
                )

            context_text = "\n".join(sections).strip()

        return {"context_text": context_text, "retrieval": retrieval}

    def generate_treatment_report(
        self,
        l1_summary: Dict[str, Any],
        all_l2_summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        l2_list: List[Dict[str, Any]] = []
        for item in all_l2_summaries:
            l2_list.append(
                {
                    "session_id": item.get("session_id", ""),
                    "topic": item.get("topic", ""),
                    "summary": item.get("summary", ""),
                }
            )

        prompt = (
            TREATMENT_PROGRESS_PROMPT
            .replace("{l1_summary}", json.dumps(l1_summary, ensure_ascii=False, indent=2))
            .replace("{l2_list}", json.dumps(l2_list, ensure_ascii=False, indent=2))
        )
        response = self._call_model(prompt)

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except Exception:
            pass

        return {
            "emotion_trend": "无明显变化趋势",
            "key_progress": [],
            "risk_note": "无",
            "treatment_phase": "进行中",
            "next_focus": "继续当前关注方向",
        }
