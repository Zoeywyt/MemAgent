from datetime import datetime
import json
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

from agents.supervisor import SupervisorAgent
from agents.summary_agent import SummaryAgent
from memory.graph_extractor import GraphExtractor
from output_store import load_user_session_summaries, load_user_trends
from utils.model_client import ChatClientProtocol, build_chat_client

if TYPE_CHECKING:
    from memory.mem0_adapter import Mem0Adapter


SYSTEM_PROMPT = (
    "你是一位经验丰富的心理咨询师，正在为同一位来访者提供长程咨询。每次会话开始前，系统会为你提供来访者的长期画像与当前咨询状态，这些是你的背景知识，请内化于心，不要在回复中复述或解释这些材料。"
    "坚守以下原则：\n"
    "1. 回应聚焦：分析并回应来访者当前主要存在的问题或者情况，要看到本质给予慢慢引导。\n"
    "2. 风格自然：使用自然流畅的专业表达，让来访者感受到平等而深入，像导师又像朋友的对话，而非诊断。\n"
    "3. 节奏克制：聊天内容保持倾听与共情，整体引导节奏不应过快，让来访者有充分的表达和思考，引导来访者进行深入的自我探索。\n"
    "4. 策略标注：回复开头用 [策略名] 标注本轮采用的主要咨询策略。\n"
    "5. 长度控制：一般回复控制在3-5句的连贯流畅段落；当来访者主动要求详细分析时可适当延长、但仍需保持清晰和重点突出。\n"
)


L1_SUMMARY_KEYS = ("主题", "背景", "会话总结")
TREATMENT_REPORT_KEYS = ("emotion_trend", "key_progress", "risk_note", "treatment_phase", "next_focus")
RETRIEVAL_DECISION_PROMPT = """你是一个咨询记忆路由器，任务只有一个：判断当前是否需要额外检索长期记忆。

请结合以下信息判断：
1. 用户当前输入。
2. 最近几轮对话。
3. 用户长期画像。
4. 当前咨询状态。

当用户明显在提到以前的人、事、时间、关系、症状变化、反复出现的问题、未解决目标，或者当前问题需要回看历史脉络时，倾向于检索。
当用户只是做当下情绪表达、一般性倾诉、简单确认、短问短答时，不需要检索。

只输出 JSON，不要解释，不要加多余文本。格式如下：
{
  "need_retrieval": true,
  "confidence": 0.0,
  "reason": "一句简短原因",
  "retrieval_focus": "如需检索，写一句搜索焦点；否则留空"
}
"""


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
        router_model_backend: Optional[str] = None,
        router_model_mode: Optional[str] = None,
        router_local_model_path: Optional[str] = None,
        router_local_base_model_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.openai_client = openai_client or build_chat_client(
            "EMPATHY_AGENT",
            backend=model_backend,
            mode=model_mode,
            local_model_path=local_model_path,
            local_base_model_path=local_base_model_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        self.summary_agent = SummaryAgent(
            model_backend=summary_model_backend or model_backend,
            model_mode=summary_model_mode or model_mode,
            local_model_path=summary_local_model_path or local_model_path,
            local_base_model_path=summary_local_base_model_path or local_base_model_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        self._mem0: Optional["Mem0Adapter"] = None
        self.graph_extractor = GraphExtractor(
            model_backend=graph_model_backend or model_backend,
            model_mode=graph_model_mode or model_mode,
            local_model_path=graph_local_model_path or local_model_path,
            local_base_model_path=graph_local_base_model_path or local_base_model_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        resolved_router_backend = router_model_backend or "gpt"
        self.router_client = build_chat_client(
            "ROUTER_AGENT",
            backend=resolved_router_backend,
            mode=router_model_mode if router_model_backend else None,
            local_model_path=router_local_model_path if router_model_backend else None,
            local_base_model_path=router_local_base_model_path if router_model_backend else None,
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_new_tokens=128,
        )
        self.supervisor: Optional[SupervisorAgent] = None
        self._supervisor_model_backend = supervisor_model_backend or model_backend
        self._supervisor_model_mode = supervisor_model_mode or model_mode
        self._supervisor_local_model_path = supervisor_local_model_path or local_model_path
        self._supervisor_local_base_model_path = supervisor_local_base_model_path or local_base_model_path
        self._api_key = api_key
        self._base_url = base_url
        self._api_model = model
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

    def _warmup_local_clients_async(self) -> None:
        clients = [
            self.openai_client,
            self.summary_agent.openai_client,
            self.graph_extractor.client,
            self.router_client,
            self.supervisor.openai_client if self.supervisor else None,
        ]
        warmups = [getattr(client, "warmup", None) for client in clients if client is not None]
        warmups = [warmup for warmup in warmups if callable(warmup)]
        if not warmups:
            return

        def _run() -> None:
            for warmup in warmups:
                try:
                    warmup()
                except Exception:
                    pass

        threading.Thread(target=_run, daemon=True, name="memagent-local-warmup").start()

    @property
    def mem0(self) -> "Mem0Adapter":
        if self._mem0 is None:
            from memory.mem0_adapter import Mem0Adapter

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
            api_key=self._api_key,
            base_url=self._base_url,
            model=self._api_model,
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
        self._warmup_local_clients_async()

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
            "【来访者长期画像】\n"
            "以下信息用于理解来访者较稳定的长期主题、背景经历与反复出现的关系/情绪模式。\n"
            f"{self._format_l1_summary(context.get('l1_summary', {}))}\n\n"
            "【当前咨询状态】\n"
            "以下信息用于判断来访者近期状态、咨询阶段、风险提示与下一步推进重点。\n"
            f"{self._format_treatment_report(context.get('treatment_report', {}))}"
        )

    def _build_system_prompt(self) -> str:
        if not self.preloaded_session_context:
            return SYSTEM_PROMPT
        return f"{SYSTEM_PROMPT}\n\n{self._build_preloaded_context_text()}"

    def _flatten_text(self, value: Any) -> str:
        if isinstance(value, dict):
            parts: List[str] = []
            for key, item in value.items():
                if item is None:
                    continue
                key_text = str(key).strip()
                if key_text:
                    parts.append(key_text)
                item_text = self._flatten_text(item)
                if item_text:
                    parts.append(item_text)
            return "\n".join(parts)
        if isinstance(value, list):
            return "\n".join(self._flatten_text(item) for item in value if item is not None)
        return str(value or "").strip()

    def _recent_history_text(self, turn_count: int = 6) -> str:
        if not self.session_messages:
            return ""
        recent = self.session_messages[-turn_count:]
        lines = []
        for idx in range(0, len(recent), 2):
            user_msg = recent[idx]
            assistant_msg = recent[idx + 1] if idx + 1 < len(recent) else None
            if isinstance(user_msg, dict):
                lines.append(f"用户：{user_msg.get('content', '')}")
            if assistant_msg and isinstance(assistant_msg, dict):
                lines.append(f"咨询师：{assistant_msg.get('content', '')}")
        return "\n".join(lines).strip()

    def _build_retrieval_decision_messages(self, user_input: str) -> List[Dict[str, str]]:
        context = self.preloaded_session_context or {}
        l1_summary = self._flatten_text(context.get("l1_summary", {}))
        treatment_report = self._flatten_text(context.get("treatment_report", {}))
        recent_history = self._recent_history_text()
        prompt = (
            "请判断本轮是否需要检索额外长期记忆。"
            "你要像一个会自我察觉的咨询智能体一样，只在确实需要补充历史上下文时再请求检索。"
            "\n\n【长期画像】\n"
            f"{l1_summary or '无'}"
            "\n\n【当前咨询状态】\n"
            f"{treatment_report or '无'}"
            "\n\n【最近对话】\n"
            f"{recent_history or '无'}"
            "\n\n【当前输入】\n"
            f"{user_input.strip()}"
            "\n\n请只输出 JSON。"
        )
        return [
            {"role": "system", "content": RETRIEVAL_DECISION_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def _parse_retrieval_decision(self, text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except Exception:
                    return {}
        return {}

    def _coerce_retrieval_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return bool(value)

    def _coerce_confidence(self, value: Any) -> Optional[float]:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        if confidence < 0:
            return 0.0
        if confidence > 1:
            return confidence / 100.0 if confidence <= 100 else 1.0
        return confidence

    def _print_retrieval_decision_debug(
        self,
        *,
        user_input: str,
        need_retrieval: bool,
        decision: Dict[str, Any],
        raw_text: str = "",
        skipped_reason: str = "",
    ) -> None:
        enabled = os.getenv("MEMAGENT_DEBUG_RETRIEVAL", "1").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return

        print("\n" + "-" * 72, flush=True)
        print("[MemAgent Router] 检索路由判决", flush=True)
        if self.current_user_id:
            print(f"user={self.current_user_id} session={self.session_id or ''}", flush=True)
        print(f"need_retrieval={need_retrieval}", flush=True)
        if skipped_reason:
            print(f"skipped_reason={skipped_reason}", flush=True)
        confidence = decision.get("confidence", "")
        reason = decision.get("reason", "")
        focus = decision.get("retrieval_focus", "")
        override_reason = decision.get("override_reason", "")
        print(f"confidence={confidence}", flush=True)
        print(f"reason={reason}", flush=True)
        if override_reason:
            print(f"override_reason={override_reason}", flush=True)
        print(f"retrieval_focus={focus}", flush=True)
        user_preview = user_input.strip().replace("\n", " ")[:300]
        print(f"user_input={user_preview}", flush=True)
        if raw_text and not decision:
            print(f"raw_router_output={raw_text.strip()[:500]}", flush=True)
        print("-" * 72 + "\n", flush=True)

    def _decide_retrieval(self, user_input: str) -> Tuple[bool, Dict[str, Any]]:
        if not self.enable_mem0_runtime:
            self._print_retrieval_decision_debug(
                user_input=user_input,
                need_retrieval=False,
                decision={},
                skipped_reason="MEMAGENT_ENABLE_MEM0_RUNTIME 未开启",
            )
            return False, {}
        if not self.supervisor:
            self._print_retrieval_decision_debug(
                user_input=user_input,
                need_retrieval=False,
                decision={},
                skipped_reason="supervisor 未初始化",
            )
            return False, {}

        messages = self._build_retrieval_decision_messages(user_input)
        try:
            decision_text, _usage = self.router_client.chat(messages=messages)
        except Exception as exc:
            self._print_retrieval_decision_debug(
                user_input=user_input,
                need_retrieval=False,
                decision={"reason": f"路由模型调用失败：{exc}"},
                skipped_reason="router_error",
            )
            return False, {}

        decision = self._parse_retrieval_decision(decision_text)
        need_retrieval = self._coerce_retrieval_bool(decision.get("need_retrieval"))
        confidence = self._coerce_confidence(decision.get("confidence"))
        if not need_retrieval and (confidence is None or confidence < 0.6):
            need_retrieval = True
            decision["override_reason"] = "router 判为不检索，但置信度低于 0.6，按保守策略执行检索"
        if "retrieval_focus" in decision and isinstance(decision.get("retrieval_focus"), str):
            decision["retrieval_focus"] = str(decision.get("retrieval_focus", "")).strip()
        decision["need_retrieval"] = need_retrieval
        self._print_retrieval_decision_debug(
            user_input=user_input,
            need_retrieval=need_retrieval,
            decision=decision,
            raw_text=decision_text,
        )
        return need_retrieval, decision

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

    def _build_response_messages(self, user_input: str) -> Tuple[List[Dict[str, str]], str, Dict[str, Any]]:
        retrieval_text = ""
        retrieval_meta: Dict[str, Any] = {
            "should_retrieve": False,
            "decision": {},
            "retrieval_query": "",
            "retrieval": {},
        }
        should_retrieve, decision = self._decide_retrieval(user_input)
        retrieval_meta["should_retrieve"] = should_retrieve
        retrieval_meta["decision"] = dict(decision)
        if should_retrieve and self.supervisor:
            retrieval_query = user_input.strip()
            retrieval_focus = str(decision.get("retrieval_focus", "") or "").strip()
            if retrieval_focus:
                retrieval_query = f"{retrieval_query}\n{retrieval_focus}"
            retrieval = self.supervisor.retrieve_context_for_response(retrieval_query)
            retrieval_meta["retrieval_query"] = retrieval_query
            retrieval_meta["retrieval"] = dict(retrieval)
            retrieval_text = retrieval.get("context_text", "")
            if retrieval_text:
                retrieval_text = (
                    "【本轮相关记忆】\n"
                    "以下内容是本轮按需检索到的历史片段与图关系，只作为理解当前输入的辅助证据；"
                    "如果与当前表达不相关，请优先回应当下。\n"
                    f"{retrieval_text}\n\n"
                )

        history_text = self._build_history_text()
        prompt = (
            f"{retrieval_text}"
            "请根据以下内容生成专业心理咨询回复。\n"
            "回复时必须在回答开始前用 [策略] 标注本轮采用的主要咨询策略。\n\n"
            "【本次会话上下文】\n"
            "历史对话：\n"
            f"{history_text}\n\n"
            "当前来访者输入：\n"
            f"{user_input}\n"
        )
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        return messages, retrieval_text.strip(), retrieval_meta

    def _finalize_response_turn(
        self,
        user_input: str,
        response_text: str,
        retrieval_context: str,
        retrieval_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
            "retrieval_context": retrieval_context,
            "retrieval_meta": retrieval_meta or {},
        }

    def generate_response(self, user_input: str) -> Dict[str, Any]:
        if self.current_user_id is None:
            raise RuntimeError("请先调用 start_session(user_id)")

        messages, retrieval_context, retrieval_meta = self._build_response_messages(user_input)
        response_text, _usage = self.openai_client.chat(messages=messages)
        return self._finalize_response_turn(user_input, response_text, retrieval_context, retrieval_meta)

    def generate_response_stream(self, user_input: str) -> Iterator[Dict[str, Any]]:
        if self.current_user_id is None:
            raise RuntimeError("请先调用 start_session(user_id)")

        messages, retrieval_context, retrieval_meta = self._build_response_messages(user_input)
        stream_chat = getattr(self.openai_client, "stream_chat", None)
        if not callable(stream_chat):
            yield self.generate_response(user_input)
            return

        response_parts: List[str] = []
        for piece in stream_chat(messages):
            if not piece:
                continue
            response_parts.append(piece)
            yield {
                "delta": piece,
                "response": "".join(response_parts),
                "done": False,
                "retrieval_context": retrieval_context,
                "retrieval_meta": retrieval_meta,
            }

        response_text = "".join(response_parts).strip()
        final_result = self._finalize_response_turn(user_input, response_text, retrieval_context, retrieval_meta)
        final_result["done"] = True
        yield final_result

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
