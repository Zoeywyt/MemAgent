from datetime import datetime
from typing import Any, Dict, List, Optional

from multi_agent.agents.supervisor import SupervisorAgent
from multi_agent.agents.summary_agent import SummaryAgent
from multi_agent.memory.graph_extractor import GraphExtractor
from multi_agent.memory.mem0_adapter import Mem0Adapter
from multi_agent.utils.openai_client import OpenAIChatClient


SYSTEM_PROMPT = (
    "你是一个共情 agent，参考用户的历史咨询摘要生成专业的心理咨询回复。"
    "回复时请在回答开始前用 [策略] 明确标注本轮采用的主要咨询策略，例如 [倾听]、[共情]、[引导]、[反思]、[探索]。"
)


class EmpathyAgent:
    def __init__(self, openai_client: Optional[OpenAIChatClient] = None):
        self.openai_client = openai_client or OpenAIChatClient()
        self.summary_agent = SummaryAgent()
        self.mem0 = Mem0Adapter()
        self.graph_extractor = GraphExtractor()
        self.supervisor: Optional[SupervisorAgent] = None
        self.session_messages: List[Dict[str, str]] = []
        self.current_user_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.session_start_time: Optional[datetime] = None

    def start_session(self, user_id: str) -> Dict[str, Optional[Dict[str, str]]]:
        self.current_user_id = user_id
        self.session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_messages = []
        self.session_start_time = datetime.now()
        self.supervisor = SupervisorAgent(user_id=user_id, mem0=self.mem0)

        l1_summary = self.summary_agent.get_user_summary(user_id)
        return {"session_id": self.session_id, "l1_summary": l1_summary}

    def _build_history_text(self) -> str:
        if not self.session_messages:
            return "暂无本次会话历史。"
        lines = []
        turn = 1
        for idx in range(0, len(self.session_messages), 2):
            user_msg = self.session_messages[idx]
            assistant_msg = self.session_messages[idx + 1] if idx + 1 < len(self.session_messages) else None
            lines.append(f"轮次{turn} 来访者：{user_msg['content']}")
            if assistant_msg:
                lines.append(f"轮次{turn} 咨询师：{assistant_msg['content']}")
            turn += 1
        return "\n".join(lines)

    def generate_response(self, user_input: str) -> Dict[str, str]:
        if self.current_user_id is None:
            raise RuntimeError("请先调用 start_session(user_id)")

        l1_summary = self.summary_agent.get_user_summary(self.current_user_id) or {}
        profile_text = "" if not l1_summary else (
            f"用户历史摘要：\n主题：{l1_summary.get('主题','')}\n背景：{l1_summary.get('背景','')}\n会话总结：{l1_summary.get('会话总结','')}\n\n"
        )

        retrieval_text = ""
        if self.supervisor:
            retrieval = self.supervisor.retrieve_context_for_response(user_input)
            retrieval_text = retrieval.get("context_text", "")
            if retrieval_text:
                retrieval_text = f"【相关检索内容】\n{retrieval_text}\n\n"

        history_text = self._build_history_text()
        prompt = (
            f"{profile_text}"
            f"{retrieval_text}"
            "请根据以下内容生成专业心理咨询回复。\n"
            "回复时必须在回答开始前用 [策略] 标注本轮采用的主要咨询策略。\n\n"
            f"【近期会话】\n{history_text}\n\n"
            f"【当前来访者输入】\n{user_input}\n"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response_text, _usage = self.openai_client.chat(messages=messages)

        self.session_messages.append({"role": "user", "content": user_input})
        self.session_messages.append({"role": "assistant", "content": response_text})
        if self.current_user_id and self.session_id:
            self.mem0.remember_l3_turn(
                user_id=self.current_user_id,
                session_id=self.session_id,
                turn_index=len(self.session_messages) // 2,
                user_input=user_input,
                assistant_response=response_text,
            )

        return {"response": response_text}

    def _format_session_for_graph(self) -> str:
        lines = []
        turn = 1
        for idx in range(0, len(self.session_messages), 2):
            user_msg = self.session_messages[idx]
            assistant_msg = self.session_messages[idx + 1] if idx + 1 < len(self.session_messages) else None
            lines.append(f"轮次{turn} user：{user_msg['content']}")
            if assistant_msg:
                lines.append(f"轮次{turn} assistant：{assistant_msg['content']}")
            turn += 1
        return "\n".join(lines)

    def end_session(self) -> Dict[str, Any]:
        if not self.session_messages or self.current_user_id is None or self.session_id is None:
            return {}

        conversation_text = self._format_session_for_graph()
        graph_data = self.graph_extractor.extract_graph(conversation_text, user_id=self.current_user_id)
        self.mem0.add_graph_data(
            self.current_user_id,
            self.session_id,
            graph_data,
            source_text=conversation_text,
        )

        l2_summary = self.summary_agent.generate_l2_summary(self.session_messages)
        self.mem0.save_l2_summary(
            self.current_user_id,
            self.session_id,
            l2_summary,
            self.session_start_time or datetime.now(),
            datetime.now(),
            len(self.session_messages) // 2,
        )

        new_l1 = self.summary_agent.update_l1_summary(self.current_user_id, l2_summary)
        all_l2 = self.mem0.get_l2_summaries(self.current_user_id, limit=100)
        report = self.supervisor.generate_treatment_report(new_l1, all_l2)
        self.mem0.save_treatment_report(self.current_user_id, report)

        self.session_messages = []
        self.session_id = None
        self.current_user_id = None
        self.session_start_time = None

        return {"graph_data": graph_data, "l1_summary": new_l1, "treatment_report": report}
