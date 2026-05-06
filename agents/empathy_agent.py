from datetime import datetime
import os
import time
from typing import Any, Dict, List, Optional

from agents.supervisor import SupervisorAgent
from agents.summary_agent import SummaryAgent
from memory.graph_extractor import GraphExtractor
from memory.mem0_adapter import Mem0Adapter
from output_store import load_user_session_summaries, load_user_trends
from utils.model_client import ChatClientProtocol, build_chat_client


SYSTEM_PROMPT = (
    "你是一个共情交互模型，需要在每次会话启动时先理解来访者的长期画像和督导师治疗进展报告，"
    "再结合当前输入生成专业、稳定且连续的心理咨询回复。"
    "你应把这些预加载资料作为临床相关知识使用，但不要机械复述，也不要让来访者感觉自己被标签化。"
    "回复时请在回答开始前用 [策略] 明确标注本轮采用的主要咨询策略，例如 [倾听]、[共情]、[引导]、[反思]、[探索]。"
)


L1_SUMMARY_KEYS = ("主题", "背景", "会话总结")
TREATMENT_REPORT_KEYS = ("emotion_trend", "key_progress", "risk_note", "treatment_phase", "next_focus")


class EmpathyAgent:
    def __init__(
        self,
        openai_client: Optional[ChatClientProtocol] = None,
        model_backend: Optional[str] = None,
        model_mode: Optional[str] = None,
        local_model_path: Optional[str] = None,
        local_base_model_path: Optional[str] = None,
        summary_model_backend: Optional[str] = None,
        summary_model_mode: Optional[str] = None,
        summary_local_model_path: Optional[str] = None,
        summary_local_base_model_path: Optional[str] = None,
        supervisor_model_backend: Optional[str] = None,
        supervisor_model_mode: Optional[str] = None,
        supervisor_local_model_path: Optional[str] = None,
        supervisor_local_base_model_path: Optional[str] = None,
        graph_model_backend: Optional[str] = None,
        graph_model_mode: Optional[str] = None,
        graph_local_model_path: Optional[str] = None,
        graph_local_base_model_path: Optional[str] = None,
    ):
        self.openai_client = openai_client or build_chat_client(
            "EMPATHY_AGENT",
            backend=model_backend,
            mode=model_mode,
            local_model_path=local_model_path,
            local_base_model_path=local_base_model_path,
        )
        self.summary_agent = SummaryAgent(
            model_backend=summary_model_backend or model_backend,
            model_mode=summary_model_mode or model_mode,
            local_model_path=summary_local_model_path or local_model_path,
            local_base_model_path=summary_local_base_model_path or local_base_model_path,
        )
        self._mem0: Optional[Mem0Adapter] = None
        self.graph_extractor = GraphExtractor(
            model_backend=graph_model_backend or model_backend,
            model_mode=graph_model_mode or model_mode,
            local_model_path=graph_local_model_path or local_model_path,
            local_base_model_path=graph_local_base_model_path or local_base_model_path,
        )
        self.supervisor: Optional[SupervisorAgent] = None
        self._supervisor_model_backend = supervisor_model_backend or model_backend
        self._supervisor_model_mode = supervisor_model_mode or model_mode
        self._supervisor_local_model_path = supervisor_local_model_path or local_model_path
        self._supervisor_local_base_model_path = supervisor_local_base_model_path or local_base_model_path
        self.session_messages: List[Dict[str, str]] = []
        self.current_user_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.session_start_time: Optional[datetime] = None
        self.preloaded_session_context: Dict[str, Any] = {}
        self.enable_mem0_runtime = os.getenv("MEMAGENT_ENABLE_MEM0_RUNTIME", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }

    @property
    def mem0(self) -> Mem0Adapter:
        if self._mem0 is None:
            self._mem0 = Mem0Adapter()
        return self._mem0

    def start_session(self, user_id: str) -> Dict[str, Any]:
        preload_timings: List[tuple[str, float]] = []

        def mark(label: str, started_at: float) -> None:
            preload_timings.append((label, time.perf_counter() - started_at))

        self.current_user_id = user_id
        self.session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_messages = []
        self.session_start_time = datetime.now()
        started = time.perf_counter()
        self.supervisor = SupervisorAgent(
            user_id=user_id,
            mem0=self._mem0,
            model_backend=self._supervisor_model_backend,
            model_mode=self._supervisor_model_mode,
            local_model_path=self._supervisor_local_model_path,
            local_base_model_path=self._supervisor_local_base_model_path,
        )
        mark("supervisor", started)

        started = time.perf_counter()
        l1_summary = self.summary_agent.get_user_summary(user_id)
        mark("l1_summary", started)
        started = time.perf_counter()
        treatment_report = self._load_latest_treatment_report_from_outputs(user_id)
        mark("treatment_outputs", started)
        self.preloaded_session_context = {
            "user_id": user_id,
            "session_id": self.session_id,
            "l1_summary": l1_summary or {},
            "treatment_report": treatment_report or {},
        }
        started = time.perf_counter()
        preloaded_context_text = self._build_preloaded_context_text()
        mark("context_text", started)

        return {
            "session_id": self.session_id,
            "l1_summary": l1_summary,
            "l2_summaries": [],
            "treatment_report": treatment_report,
            "preloaded_context_text": preloaded_context_text,
            "preload_timings": preload_timings,
        }

    def _load_l2_summaries_from_outputs(self, user_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        payload = load_user_session_summaries(user_id)
        sessions = payload.get("sessions", [])
        if not isinstance(sessions, list):
            return []

        items: List[Dict[str, Any]] = []
        for item in sessions:
            if not isinstance(item, dict):
                continue
            summary = item.get("summary", "")
            topic = ""
            summary_text = ""
            if isinstance(summary, dict):
                topic = str(summary.get("topic") or summary.get("主题") or "")
                background = str(summary.get("background") or summary.get("背景") or "")
                main_summary = str(summary.get("summary") or summary.get("总结") or summary.get("会话总结") or "")
                summary_text = "\n".join(part for part in [background, main_summary] if part).strip()
            else:
                summary_text = str(summary or "")
                if "主题" in summary_text:
                    first_line = summary_text.splitlines()[0].strip()
                    topic = first_line.replace("主题：", "").replace("主题:", "").strip()
            items.append(
                {
                    "session_id": item.get("session_id", ""),
                    "topic": topic,
                    "summary": summary_text,
                    "start_time": item.get("archived_at") or item.get("generated_at") or "",
                }
            )

        items.sort(key=lambda value: value.get("start_time", ""), reverse=True)
        return items[:limit]

    def _load_latest_treatment_report_from_outputs(self, user_id: str) -> Optional[Dict[str, Any]]:
        payload = load_user_trends(user_id)
        reports = payload.get("reports", [])
        if not isinstance(reports, list):
            return None

        latest: Optional[Dict[str, Any]] = None
        latest_time = ""
        for item in reports:
            if not isinstance(item, dict):
                continue
            report = item.get("report")
            if not isinstance(report, dict):
                continue
            generated_at = str(item.get("generated_at") or item.get("archived_at") or "")
            if latest is None or generated_at >= latest_time:
                latest = report
                latest_time = generated_at
        return latest

    def _format_l1_summary(self, l1_summary: Dict[str, Any]) -> str:
        if not l1_summary:
            return "暂无长期画像摘要。"
        lines = []
        for key in L1_SUMMARY_KEYS:
            value = l1_summary.get(key, "")
            if value:
                lines.append(f"{key}：{value}")
        return "\n".join(lines) if lines else "暂无长期画像摘要。"

    def _format_l2_summaries(self, l2_summaries: List[Dict[str, Any]]) -> str:
        if not l2_summaries:
            return "暂无历史会话摘要。"
        lines = []
        for index, item in enumerate(l2_summaries, start=1):
            session_id = item.get("session_id", f"历史会话{index}")
            topic = item.get("topic", "")
            summary = item.get("summary", "")
            prefix = f"{index}. {session_id}"
            if topic:
                prefix += f" | {topic}"
            lines.append(f"{prefix}\n{summary}")
        return "\n".join(lines)

    def _format_treatment_report(self, report: Dict[str, Any]) -> str:
        if not report:
            return "暂无督导师治疗进展报告。"
        labels = {
            "emotion_trend": "情绪变化轨迹",
            "key_progress": "关键进展",
            "risk_note": "风险评估",
            "treatment_phase": "治疗阶段",
            "next_focus": "下一阶段重点",
        }
        lines = []
        for key in TREATMENT_REPORT_KEYS:
            value = report.get(key)
            if not value:
                continue
            if isinstance(value, list):
                value_text = "；".join(str(item) for item in value if item)
            else:
                value_text = str(value)
            if value_text:
                lines.append(f"{labels[key]}：{value_text}")
        return "\n".join(lines) if lines else "暂无督导师治疗进展报告。"

    def _build_preloaded_context_text(self) -> str:
        context = self.preloaded_session_context or {}
        return (
            "【会话启动预加载资料】\n"
            "以下资料已在本次会话开始前自动加载，用于建立对来访者整体情况的连续理解。\n\n"
            "【长期画像记忆 / L1】\n"
            f"{self._format_l1_summary(context.get('l1_summary', {}))}\n\n"
            "【督导师最新治疗进展报告】\n"
            f"{self._format_treatment_report(context.get('treatment_report', {}))}"
        )

    def _build_system_prompt(self) -> str:
        if not self.preloaded_session_context:
            return SYSTEM_PROMPT
        return f"{SYSTEM_PROMPT}\n\n{self._build_preloaded_context_text()}"

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

        retrieval_text = ""
        if self.enable_mem0_runtime and self.supervisor:
            retrieval = self.supervisor.retrieve_context_for_response(user_input)
            retrieval_text = retrieval.get("context_text", "")
            if retrieval_text:
                retrieval_text = f"【本轮相关检索补充】\n{retrieval_text}\n\n"

        history_text = self._build_history_text()
        prompt = (
            f"{retrieval_text}"
            "请根据以下内容生成专业心理咨询回复。\n"
            "回复时必须在回答开始前用 [策略] 标注本轮采用的主要咨询策略。\n\n"
            f"【近期会话】\n{history_text}\n\n"
            f"【当前来访者输入】\n{user_input}\n"
        )

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        response_text, _usage = self.openai_client.chat(messages=messages)

        self.session_messages.append({"role": "user", "content": user_input})
        self.session_messages.append({"role": "assistant", "content": response_text})
        l3_memory_result: Dict[str, Any] = {"facts": [], "actions": [], "stored": 0}
        if self.enable_mem0_runtime and self.current_user_id and self.session_id:
            l3_memory_result = self.mem0.remember_l3_turn(
                user_id=self.current_user_id,
                session_id=self.session_id,
                turn_index=len(self.session_messages) // 2,
                user_input=user_input,
                assistant_response=response_text,
            )

        return {
            "response": response_text,
            "l3_memory_result": l3_memory_result,
            "retrieval_context": retrieval_text.strip(),
        }

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
        if self.enable_mem0_runtime:
            self.mem0.add_graph_data(
                self.current_user_id,
                self.session_id,
                graph_data,
                source_text=conversation_text,
            )

        l2_summary = self.summary_agent.generate_l2_summary(self.session_messages)
        if self.enable_mem0_runtime:
            self.mem0.save_l2_summary(
                self.current_user_id,
                self.session_id,
                l2_summary,
                self.session_start_time or datetime.now(),
                datetime.now(),
                len(self.session_messages) // 2,
            )

        new_l1 = self.summary_agent.update_l1_summary(self.current_user_id, l2_summary)
        all_l2 = self._load_l2_summaries_from_outputs(self.current_user_id, limit=100)
        all_l2.insert(
            0,
            {
                "session_id": self.session_id,
                "topic": l2_summary.get("topic", ""),
                "summary": l2_summary.get("summary", ""),
                "start_time": datetime.now().isoformat(),
            },
        )
        report = self.supervisor.generate_treatment_report(new_l1, all_l2)
        if self.enable_mem0_runtime:
            self.mem0.save_treatment_report(self.current_user_id, report)

        self.session_messages = []
        self.session_id = None
        self.current_user_id = None
        self.session_start_time = None
        self.preloaded_session_context = {}

        return {"graph_data": graph_data, "l2_summary": l2_summary, "l1_summary": new_l1, "treatment_report": report}
