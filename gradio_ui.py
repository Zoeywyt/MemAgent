from __future__ import annotations

import json
import html
import os
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

CURRENT_DIR = Path(__file__).resolve().parent
TESTS_DIR = CURRENT_DIR / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from agents.empathy_agent import EmpathyAgent
from memory.user_summary_store import UserSummaryStore
from output_store import (
    CONSULTING_REPORTS_DIR,
    USER_REGISTRY_PATH,
    consulting_report_html_path,
    consulting_report_json_path,
    ensure_output_dirs,
    load_user_sessions,
    migrate_legacy_session_tree,
    record_session_outputs,
    load_user_session_summaries,
    sessions_summaries_path,
    sessions_path,
    load_user_trends,
    trends_path,
    whole_summary_path,
    WHOLE_SUMMARIES_DIR,
    safe_user_id,
)
from test_case import build_result_bundle


ensure_output_dirs()
REPORT_DIR = CONSULTING_REPORTS_DIR

AGENT_BACKEND_CHOICES = [
    ("GPT API", "gpt"),
    ("Qwen2.5-3B + 3B LoRA", "qwen3b"),
    ("Qwen2.5-7B + 7B LoRA", "qwen7b"),
]

SUMMARY_BACKEND_CHOICES = [
    ("GPT API", "gpt"),
    ("Qwen2.5-3B + 3B LoRA", "qwen3b"),
]


def load_user_registry() -> Dict[str, Any]:
    if not USER_REGISTRY_PATH.exists():
        return {"users": []}
    try:
        return json.loads(USER_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"users": []}


def save_user_registry(registry: Dict[str, Any]) -> None:
    USER_REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize_user_id(raw_value: str) -> str:
    text = (raw_value or "").strip()
    if not text:
        raise gr.Error("请输入用户名或用户 ID。")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        raise gr.Error("用户 ID 非法，请换一个名称。")
    return text


def collect_known_user_ids() -> List[str]:
    user_ids = set()
    registry = load_user_registry()
    for item in registry.get("users", []):
        if isinstance(item, dict) and item.get("user_id"):
            user_ids.add(str(item["user_id"]))

    summary_dir = Path("multi_agent_user_summaries")
    if summary_dir.exists():
        for path in summary_dir.glob("*_summary.json"):
            user_ids.add(path.name[: -len("_summary.json")])

    if WHOLE_SUMMARIES_DIR.exists():
        for path in WHOLE_SUMMARIES_DIR.glob("*_summary.json"):
            user_ids.add(path.name[: -len("_summary.json")])

    return sorted(user_ids)


def ensure_seed_users() -> None:
    registry = load_user_registry()
    known = {item.get("user_id", "") for item in registry.get("users", []) if isinstance(item, dict)}
    changed = False
    for user_id in collect_known_user_ids():
        if user_id and user_id not in known:
            registry.setdefault("users", []).append(
                {
                    "user_id": user_id,
                    "display_name": user_id,
                    "created_at": datetime.now().isoformat(),
                    "last_login_at": "",
                    "source": "seeded",
                }
            )
            changed = True
    if changed or not USER_REGISTRY_PATH.exists():
        save_user_registry(registry)


def find_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    registry = load_user_registry()
    for item in registry.get("users", []):
        if isinstance(item, dict) and item.get("user_id") == user_id:
            return item
    return None


def register_user(display_name: str) -> str:
    user_id = sanitize_user_id(display_name)
    if find_user_profile(user_id):
        raise gr.Error(f"用户 `{user_id}` 已存在，请直接登录。")

    registry = load_user_registry()
    registry.setdefault("users", []).append(
        {
            "user_id": user_id,
            "display_name": display_name.strip() or user_id,
            "created_at": datetime.now().isoformat(),
            "last_login_at": "",
            "source": "manual_register",
        }
    )
    save_user_registry(registry)

    summary_store = UserSummaryStore()
    if summary_store.get_user_summary(user_id) is None:
        summary_store.save_user_summary(
            user_id,
            {"主题": "未知", "背景": "用户信息不足", "会话总结": "暂无对话历史"},
        )
    return user_id


def build_agent(agent_backend: str, summary_backend: str, supervisor_backend: str, graph_backend: str) -> EmpathyAgent:
    return EmpathyAgent(
        model_backend=agent_backend,
        summary_model_backend=summary_backend,
        supervisor_model_backend=supervisor_backend,
        graph_model_backend=graph_backend,
    )


def empty_state() -> Dict[str, Any]:
    return {
        "authenticated": False,
        "user_id": "",
        "display_name": "",
        "agent": None,
        "chatbot": [],
        "l3_records": [],
        "current_session_id": "",
        "preloaded_session_context": {},
        "config": {},
        "agent_ready": False,
        "hub_section": "overview",
        "history_session_id": "",
    }


def login_user(
    selected_user_id: str,
    typed_user_id: str,
    agent_backend: str,
    summary_backend: str,
    supervisor_backend: str,
    graph_backend: str,
) -> Tuple[Dict[str, Any], str, dict, dict, str, str, dict, str, dict]:
    selected_user_id = (selected_user_id or "").strip()
    typed_user_id = (typed_user_id or "").strip()
    user_id = sanitize_user_id(typed_user_id) if typed_user_id else selected_user_id
    if not user_id:
        raise gr.Error("请选择已有用户或输入用户名称。")

    profile = find_user_profile(user_id)
    if not profile:
        user_id = register_user(user_id)
        profile = find_user_profile(user_id)

    now = datetime.now().isoformat()
    registry = load_user_registry()
    for item in registry.get("users", []):
        if isinstance(item, dict) and item.get("user_id") == user_id:
            item["last_login_at"] = now
            item.setdefault("display_name", user_id)
    save_user_registry(registry)

    state = empty_state()
    state["authenticated"] = True
    state["user_id"] = user_id
    state["display_name"] = str((profile.get("display_name") if profile else None) or user_id)
    state["config"] = {
        "agent_backend": agent_backend,
        "summary_backend": summary_backend,
        "supervisor_backend": supervisor_backend,
        "graph_backend": graph_backend,
    }
    memory_html = build_session_memory_panel(user_id, state["display_name"])
    choices = session_choices(user_id)
    selected_session = choices[-1] if choices else ""
    hub_html, _, meta = render_consultation_hub(user_id, "overview", selected_session)
    state["history_session_id"] = meta.get("history_session_id", selected_session)
    return (
        state,
        f"登录成功：{state['display_name']} ({user_id})",
        gr.update(visible=False),
        gr.update(visible=True),
        state["display_name"],
        memory_html,
        gr.update(choices=choices, value=selected_session, visible=False),
        hub_html,
        gr.update(visible=False),
    )


def register_and_refresh(display_name: str) -> Tuple[str, dict]:
    user_id = register_user(display_name)
    return f"注册成功：{user_id}", gr.update(choices=collect_known_user_ids(), value=user_id)


def logout_user() -> Tuple[Dict[str, Any], str, str, str, dict, dict, dict, str, str, str, str, str, dict, str]:
    return (
        empty_state(),
        "已退出登录。",
        "",
        "",
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        "",
        "",
        "",
        "",
        gr.update(choices=[], value="", visible=False),
        "",
    )


def ensure_agent_ready(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录成功后再进入会话。")
    if state.get("agent_ready") and state.get("agent") is not None:
        return state

    timings: List[Tuple[str, float]] = []

    def mark(label: str, started_at: float) -> None:
        timings.append((label, time.perf_counter() - started_at))

    config = state.get("config", {})
    started = time.perf_counter()
    agent = build_agent(
        config.get("agent_backend", "gpt"),
        config.get("summary_backend", "gpt"),
        config.get("supervisor_backend", "gpt"),
        config.get("graph_backend", "gpt"),
    )
    mark("build_agent", started)
    started = time.perf_counter()
    session_info = agent.start_session(state["user_id"])
    mark("start_session", started)
    for label, elapsed in session_info.get("preload_timings", []):
        timings.append((f"start.{label}", elapsed))
    started = time.perf_counter()
    state["agent"] = agent
    state["current_session_id"] = session_info.get("session_id", "")
    state["preloaded_session_context"] = {
        "l1_summary": session_info.get("l1_summary"),
        "treatment_report": session_info.get("treatment_report"),
        "preloaded_context_text": session_info.get("preloaded_context_text", ""),
    }
    state["agent_ready"] = True
    state["agent_ready_timings"] = timings
    return state


def format_agent_ready_timings(state: Dict[str, Any]) -> str:
    timings = state.get("agent_ready_timings", [])
    if not timings:
        return ""
    detail = "，".join(f"{label}={elapsed:.2f}s" for label, elapsed in timings)
    return f"预加载耗时：{detail}"


def _html_escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def render_json_value(value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return '<div class="empty-card">暂无内容</div>'
        rows = []
        for key, item in value.items():
            rows.append(
                '<div class="json-row">'
                f'<div class="json-key">{_html_escape(key)}</div>'
                f'<div class="json-value">{render_json_value(item)}</div>'
                "</div>"
            )
        return "".join(rows)
    if isinstance(value, list):
        if not value:
            return '<div class="empty-card">暂无内容</div>'
        items = "".join(
            f'<li><span class="list-index">{index}</span><div class="list-content">{render_json_value(item)}</div></li>'
            for index, item in enumerate(value, start=1)
        )
        return f'<ol class="json-list">{items}</ol>'
    text = _html_escape(value).replace("\n", "<br>")
    return f'<div class="json-text">{text or "暂无内容"}</div>'


REPORT_FIELD_LABELS = {
    "emotion_trend": "情绪变化轨迹",
    "key_progress": "关键进展",
    "risk_note": "风险评估",
    "treatment_phase": "咨询阶段",
    "next_focus": "下一阶段重点",
}


def localize_report_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {REPORT_FIELD_LABELS.get(str(key), str(key)): localize_report_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [localize_report_payload(item) for item in payload]
    return payload


def render_document_card(title: str, payload: Any, *, subtitle: str = "", path: str = "") -> str:
    subtitle_html = f'<div class="doc-subtitle">{_html_escape(subtitle)}</div>' if subtitle else ""
    return (
        '<section class="doc-card">'
        f'<div class="doc-title">{_html_escape(title)}</div>'
        f"{subtitle_html}"
        f'<div class="doc-body">{render_json_value(payload)}</div>'
        "</section>"
    )


def render_fold_panel(title: str, payload: Any, subtitle: str = "") -> str:
    return (
        '<details class="fold-card">'
        f'<summary>{_html_escape(title)}</summary>'
        '<div class="fold-card-body">'
        f'{render_document_card(title, payload, subtitle=subtitle)}'
        "</div>"
        "</details>"
    )


def build_session_memory_panel(user_id: str, display_name: str) -> str:
    profile = read_json_payload(whole_summary_path(user_id), {})
    report_payload, _ = latest_treatment_report_payload(user_id)
    return (
        '<div class="fold-grid">'
        f'{render_fold_panel("我的画像", profile, subtitle=display_name)}'
        f'{render_fold_panel("目前督导师报告", localize_report_payload(report_payload), subtitle=display_name)}'
        '</div>'
    )


def session_choices(user_id: str) -> List[str]:
    doc = load_user_sessions(user_id)
    sessions = doc.get("sessions", []) if isinstance(doc.get("sessions", []), list) else []
    choices = [str(item.get("session_id")) for item in sessions if isinstance(item, dict) and item.get("session_id")]
    return choices


def render_history_session_detail(user_id: str, session_id: str) -> str:
    doc = load_user_sessions(user_id)
    sessions = doc.get("sessions", []) if isinstance(doc.get("sessions", []), list) else []
    selected = None
    for item in sessions:
        if isinstance(item, dict) and str(item.get("session_id", "")) == str(session_id):
            selected = item
            break
    if not selected:
        return render_document_card("历史会话", {"提示": "请选择一个会话编号。"})

    turns = []
    for turn in selected.get("turns", []) or []:
        if isinstance(turn, dict):
            turns.append({"用户": turn.get("user", ""), "咨询师": turn.get("assistant", "")})
    body = {
        "会话编号": selected.get("session_id", ""),
        "归档时间": selected.get("archived_at", ""),
        "对话轮数": selected.get("turn_count", len(turns)),
        "历史交互": turns,
    }
    return render_document_card("历史会话", body, subtitle=f"会话编号：{selected.get('session_id', '')}")


def render_consultation_hub(user_id: str, section: str = "overview", session_id: str = "") -> Tuple[str, str, Dict[str, Any]]:
    profile_html = render_document_card("我的画像", read_json_payload(whole_summary_path(user_id), {}))
    report_payload, _ = latest_treatment_report_payload(user_id)
    _ = render_document_card("目前督导师报告", localize_report_payload(report_payload))
    summary_doc = load_user_session_summaries(user_id)
    full_report_payload = {
        "final_l1_summary": read_json_payload(whole_summary_path(user_id), {}),
        "session_results": collect_archived_session_results(user_id),
        "all_l2_summaries": summary_doc.get("sessions", []) if isinstance(summary_doc, dict) else [],
    }
    choices = session_choices(user_id)
    latest_history = session_id or (choices[-1] if choices else "")
    if section == "profile":
        return profile_html, "已打开长期记忆。", {"history_visible": False, "history_session_id": latest_history}
    if section == "report":
        return render_current_full_report(user_id, full_report_payload, ""), "已打开阶段摘要。", {"history_visible": False, "history_session_id": latest_history}
    if section == "history":
        return render_history_session_detail(user_id, latest_history), "已打开报告追踪。", {"history_visible": True, "history_session_id": latest_history}
    return (
        render_document_card(
            "我的咨询",
            {
                "连续咨询": "返回聊天界面继续对话",
                "长期记忆": "查看我的画像",
                "阶段摘要": "查看完整报告",
                "报告追踪": "查看历史会话",
            },
        ),
        "已打开我的咨询。",
        {"history_visible": False, "history_session_id": latest_history},
    )


def read_json_payload(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def resource_button_updates(active: str) -> Tuple[dict, dict, dict, dict]:
    names = ("profile", "supervisor", "report", "history")
    return tuple(gr.update(variant="primary" if name == active else "secondary") for name in names)


def latest_session_summary_payload(user_id: str) -> Tuple[Any, str]:
    path = sessions_summaries_path(user_id)
    doc = load_user_session_summaries(user_id)
    sessions = doc.get("sessions", []) if isinstance(doc.get("sessions", []), list) else []
    if not sessions:
        return {}, str(path.resolve())
    latest = max(
        (item for item in sessions if isinstance(item, dict)),
        key=lambda item: (item.get("id", 0), str(item.get("archived_at", ""))),
        default={},
    )
    return latest.get("summary", latest), str(path.resolve())


def latest_treatment_report_payload(user_id: str) -> Tuple[Any, str]:
    path = trends_path(user_id)
    doc = load_user_trends(user_id)
    reports = doc.get("reports", []) if isinstance(doc.get("reports", []), list) else []
    if not reports:
        return {}, str(path.resolve())
    latest = max(
        (item for item in reports if isinstance(item, dict)),
        key=lambda item: (item.get("id", 0), str(item.get("generated_at", ""))),
        default={},
    )
    return latest.get("report", latest), str(path.resolve())


def latest_consulting_report_html(user_id: str) -> Tuple[str, str]:
    safe_id = safe_user_id(user_id)
    candidates = sorted(
        CONSULTING_REPORTS_DIR.glob(f"{safe_id}_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return "", ""
    path = candidates[0]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "", str(path.resolve())
    return render_current_full_report(user_id, payload, str(path.resolve())), str(path.resolve())


def normalize_summary_text(summary: Any) -> Dict[str, Any]:
    if isinstance(summary, dict):
        return summary
    text = str(summary or "").strip()
    if not text:
        return {}
    return {"摘要": text}


def session_report_cards(user_id: str) -> str:
    summaries_doc = load_user_session_summaries(user_id)
    trends_doc = load_user_trends(user_id)
    summaries = summaries_doc.get("sessions", []) if isinstance(summaries_doc.get("sessions", []), list) else []
    reports = trends_doc.get("reports", []) if isinstance(trends_doc.get("reports", []), list) else []
    report_by_session = {
        str(item.get("session_id", "")): item.get("report", {})
        for item in reports
        if isinstance(item, dict)
    }
    cards: List[str] = []
    for item in summaries:
        if not isinstance(item, dict):
            continue
        session_id = str(item.get("session_id", ""))
        summary = normalize_summary_text(item.get("summary"))
        report = localize_report_payload(report_by_session.get(session_id, {}))
        cards.append(
            '<section class="doc-card session-report-card">'
            f'<div class="doc-title">{_html_escape(session_id or "Session")}</div>'
            f'<div class="doc-subtitle">记忆落库时间：{_html_escape(item.get("archived_at", ""))}</div>'
            '<div class="two-report-cols">'
            f'<div>{render_document_card("会话记忆摘要", summary)}</div>'
            f'<div>{render_document_card("本次督导师报告", report)}</div>'
            '</div>'
            '</section>'
        )
    if not cards:
        return render_document_card("Session 记忆与督导师报告", {})
    return "".join(cards)


def render_current_full_report(user_id: str, payload: Dict[str, Any], path: str) -> str:
    l1 = payload.get("final_l1_summary") or {}
    sessions = payload.get("session_results", []) if isinstance(payload.get("session_results", []), list) else []
    l2 = payload.get("all_l2_summaries", []) if isinstance(payload.get("all_l2_summaries", []), list) else []
    overview = {
        "用户": user_id,
        "已归档会话数": len(sessions),
        "会话摘要数": len(l2),
    }
    return (
        '<div class="full-report-shell">'
        f'{render_document_card("完整报告概览", overview)}'
        f'{render_document_card("长期画像", l1)}'
        f'{session_report_cards(user_id)}'
        '</div>'
    )


def send_message(user_input: str, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any], List[Dict[str, str]], str]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录成功后再进入会话。")
    if not user_input.strip():
        return "", state, state.get("chatbot", []), "请输入消息后再发送。"

    state = ensure_agent_ready(state)
    agent: EmpathyAgent = state["agent"]
    result = agent.generate_response(user_input.strip())

    chatbot = list(state.get("chatbot", []))
    chatbot.append({"role": "user", "content": user_input.strip()})
    chatbot.append({"role": "assistant", "content": result["response"]})
    state["chatbot"] = chatbot
    l3_records = list(state.get("l3_records", []))
    l3_records.append(
        {
            "turn_index": len(l3_records) + 1,
            "user": user_input.strip(),
            "assistant": result["response"],
            "l3_memory_result": result.get("l3_memory_result", {}),
            "retrieval_context": result.get("retrieval_context", ""),
        }
    )
    state["l3_records"] = l3_records

    timing_text = format_agent_ready_timings(state)
    status = f"已回复。当前会话：{state.get('current_session_id', '')}"
    if timing_text:
        status = f"{status}\n{timing_text}"
    return "", state, chatbot, status


def collect_archived_session_results(user_id: str) -> List[Dict[str, Any]]:
    session_results: List[Dict[str, Any]] = []
    payload = load_user_sessions(user_id)
    for session in payload.get("sessions", []):
        chat_pairs = [
            {"user": turn.get("user", ""), "assistant": turn.get("assistant", "")}
            for turn in session.get("turns", [])
        ]
        session_results.append(
            {
                "session_label": session.get("session_id", ""),
                "run_session_id": session.get("run_session_id", session.get("session_id", "")),
                "chat_pairs": chat_pairs,
                "l3_records": [],
                "end_result": {},
            }
        )
    return session_results


def finalize_session_if_needed(state: Dict[str, Any], *, restart_session: bool = True) -> Dict[str, Any]:
    state = ensure_agent_ready(state)
    if not state.get("chatbot"):
        return state

    agent: EmpathyAgent = state["agent"]
    end_result = agent.end_session()

    chat_pairs = []
    for idx in range(0, len(state["chatbot"]), 2):
        user_msg = state["chatbot"][idx]
        assistant_msg = state["chatbot"][idx + 1] if idx + 1 < len(state["chatbot"]) else {"content": ""}
        if user_msg.get("role") == "user":
            chat_pairs.append({"user": user_msg.get("content", ""), "assistant": assistant_msg.get("content", "")})
    output_paths = record_session_outputs(
        user_id=state["user_id"],
        display_name=state["display_name"],
        run_session_id=state["current_session_id"],
        chat_pairs=chat_pairs,
        l2_summary=end_result.get("l2_summary", {}),
        treatment_report=end_result.get("treatment_report", {}),
        l1_summary=end_result.get("l1_summary", {}),
        archived_at=datetime.now().isoformat(),
    )
    state["last_output_paths"] = output_paths

    state["chatbot"] = []
    state["l3_records"] = []
    if restart_session:
        session_info = agent.start_session(state["user_id"])
        state["current_session_id"] = session_info.get("session_id", "")
        state["preloaded_session_context"] = {
            "l1_summary": session_info.get("l1_summary"),
            "treatment_report": session_info.get("treatment_report"),
            "preloaded_context_text": session_info.get("preloaded_context_text", ""),
        }
    else:
        state["current_session_id"] = ""
        state["preloaded_session_context"] = {}
        state["agent_ready"] = False
        state["agent"] = None
    return state


def ensure_report_bundle(state: Dict[str, Any], prefix: str = "manual") -> Dict[str, Any]:
    state = finalize_session_if_needed(state, restart_session=True)
    state = ensure_agent_ready(state)
    agent: EmpathyAgent = state["agent"]
    archived_session_results = collect_archived_session_results(state["user_id"])
    summary_doc = load_user_session_summaries(state["user_id"])
    trends_doc = load_user_trends(state["user_id"])
    trend_reports = trends_doc.get("reports", []) if isinstance(trends_doc.get("reports", []), list) else []
    latest_report = {}
    if trend_reports:
        latest_report = max(
            trend_reports,
            key=lambda item: str(item.get("generated_at", "")) if isinstance(item, dict) else "",
        ).get("report", {})
    bundle = build_result_bundle(
        agent=agent,
        user_id=state["user_id"],
        session_results=archived_session_results,
        completed_sessions=[item.get("session_label", "") for item in archived_session_results],
        status="active",
        include_retrieval_examples=False,
    )
    bundle["all_l2_summaries"] = summary_doc.get("sessions", [])
    bundle["latest_treatment_report"] = latest_report
    result_path = consulting_report_json_path(state["user_id"], prefix=prefix)
    result_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    html_text = render_current_full_report(state["user_id"], bundle, str(result_path.resolve()))
    html_path = consulting_report_html_path(state["user_id"], prefix=prefix)
    html_path.write_text(html_text, encoding="utf-8")
    bundle["_html_text"] = html_text
    bundle["_html_path"] = str(html_path.resolve())
    return bundle


def show_profile(state: Dict[str, Any]) -> Tuple[str, str, dict, dict, dict, dict]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    payload = read_json_payload(whole_summary_path(state["user_id"]), {})
    html_text = render_document_card(
        "我的画像",
        payload,
        subtitle=state["display_name"],
    )
    return html_text, "已加载我的画像。", *resource_button_updates("profile")


def show_supervisor_report(state: Dict[str, Any]) -> Tuple[str, str, dict, dict, dict, dict]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    payload, _ = latest_treatment_report_payload(state["user_id"])
    html_text = render_document_card("目前督导师报告", localize_report_payload(payload), subtitle=state["display_name"])
    return html_text, "已加载目前督导师报告。", *resource_button_updates("supervisor")


def show_full_report(state: Dict[str, Any]) -> Tuple[str, str, dict, dict, dict, dict]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    html_text, _ = latest_consulting_report_html(state["user_id"])
    if html_text:
        return html_text, "已加载完整报告。", *resource_button_updates("report")
    summary_doc = load_user_session_summaries(state["user_id"])
    virtual_payload = {
        "final_l1_summary": read_json_payload(whole_summary_path(state["user_id"]), {}),
        "session_results": collect_archived_session_results(state["user_id"]),
        "all_l2_summaries": summary_doc.get("sessions", []) if isinstance(summary_doc, dict) else [],
    }
    html_text = render_current_full_report(state["user_id"], virtual_payload, "")
    return html_text, "已加载完整报告。", *resource_button_updates("report")


def render_history_sessions(user_id: str) -> str:
    doc = load_user_sessions(user_id)
    sessions = doc.get("sessions", []) if isinstance(doc.get("sessions", []), list) else []
    if not sessions:
        return render_document_card("历史会话", {"提示": "暂无历史会话。"})

    cards: List[str] = []
    for index, session in enumerate(sessions, start=1):
        if not isinstance(session, dict):
            continue
        turns = []
        for turn in session.get("turns", []) or []:
            if not isinstance(turn, dict):
                continue
            turns.append(
                {
                    "用户": turn.get("user", ""),
                    "咨询师": turn.get("assistant", ""),
                }
            )
        body = {
            "会话编号": session.get("session_id") or f"Session{index}",
            "归档时间": session.get("archived_at", ""),
            "对话轮数": session.get("turn_count", len(turns)),
            "历史交互": turns,
        }
        cards.append(render_document_card(f"历史会话 {index}", body))
    return '<div class="history-shell">' + "".join(cards) + "</div>"


def show_history_sessions(state: Dict[str, Any]) -> Tuple[str, str, dict, dict, dict, dict]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    return render_history_sessions(state["user_id"]), "已加载历史会话。", *resource_button_updates("history")


def open_chat_page(state: Dict[str, Any]) -> Tuple[dict, dict, str]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    return gr.update(visible=True), gr.update(visible=False), "已返回咨询对话。"


def open_consultation_hub(state: Dict[str, Any]) -> Tuple[dict, dict, str, dict, str]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    choices = session_choices(state["user_id"])
    selected = state.get("history_session_id") or (choices[-1] if choices else "")
    html_text, status, meta = render_consultation_hub(state["user_id"], "overview", selected)
    state["history_session_id"] = meta.get("history_session_id", selected)
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        html_text,
        gr.update(choices=choices, value=state["history_session_id"], visible=False),
        status,
    )


def show_hub_profile(state: Dict[str, Any]) -> Tuple[str, dict, str]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    html_text, status, _ = render_consultation_hub(state["user_id"], "profile", state.get("history_session_id", ""))
    return html_text, gr.update(visible=False), status


def show_hub_full_report(state: Dict[str, Any]) -> Tuple[str, dict, str]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    html_text, status, _ = render_consultation_hub(state["user_id"], "report", state.get("history_session_id", ""))
    return html_text, gr.update(visible=False), status


def show_hub_history(state: Dict[str, Any]) -> Tuple[Dict[str, Any], str, dict, str]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    choices = session_choices(state["user_id"])
    selected = state.get("history_session_id") or (choices[-1] if choices else "")
    state["history_session_id"] = selected
    html_text, status, _ = render_consultation_hub(state["user_id"], "history", selected)
    return state, html_text, gr.update(choices=choices, value=selected, visible=True), status


def select_history_session(session_id: str, state: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    state["history_session_id"] = session_id or ""
    return state, render_history_session_detail(state["user_id"], state["history_session_id"]), "已切换历史会话。"


def end_session(state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]], str, str, str, str]:
    if not state.get("chatbot"):
        memory_html = build_session_memory_panel(state["user_id"], state["display_name"]) if state.get("user_id") else ""
        return state, [], "当前没有新的对话内容，本次不会写入记忆或会话记录。", "", "", memory_html
    state = finalize_session_if_needed(state, restart_session=True)
    output_paths = state.get("last_output_paths", {})
    summary_payload, summary_path = latest_session_summary_payload(state["user_id"])
    report_payload, report_path = latest_treatment_report_payload(state["user_id"])
    summary_html = render_document_card("本次会话总结", summary_payload, path=summary_path)
    report_html = render_document_card("本次督导师报告", localize_report_payload(report_payload), path=report_path)
    memory_html = build_session_memory_panel(state["user_id"], state["display_name"])
    return (
        state,
        [],
        "当前会话已结束并拆分归档。",
        summary_html,
        report_html,
        memory_html,
    )


def warmup_ui_callback() -> None:
    return None


APP_CSS = """
:root {
  --blue-950: #0b2f66;
  --blue-800: #1456c9;
  --blue-700: #1677ff;
  --blue-500: #29a8ff;
  --blue-300: #7bd8ff;
  --blue-100: #eaf6ff;
  --sky-50: #f6fbff;
  --ink: #13233a;
  --muted: #607289;
  --line: #d7e7f7;
  --panel: rgba(255, 255, 255, 0.96);
  --shadow: 0 18px 42px rgba(32, 96, 170, 0.12);
}

.gradio-container {
  min-height: 100vh;
  color: var(--ink);
  background:
    linear-gradient(90deg, rgba(22, 119, 255, 0.035) 1px, transparent 1px),
    linear-gradient(0deg, rgba(22, 119, 255, 0.035) 1px, transparent 1px),
    linear-gradient(180deg, #fbfdff 0%, #f4faff 48%, #ffffff 100%);
  background-size: 32px 32px, 32px 32px, 100% 100%;
  animation: workspaceGrid 34s linear infinite;
}

.gradio-container::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  opacity: 0.9;
  background:
    linear-gradient(180deg, rgba(22, 119, 255, 0.08), transparent 18%),
    linear-gradient(90deg, rgba(41, 168, 255, 0.12), transparent 28%, transparent 72%, rgba(41, 168, 255, 0.08));
}

.gradio-container .main {
  background: transparent;
  padding-top: 18px;
}

.app-shell,
.auth-shell,
.chat-shell {
  max-width: 1180px;
  margin: 0 auto;
}

.hero-layout {
  display: grid !important;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 18px;
  margin: 4px auto 14px;
  padding: 24px 28px;
  border: 1px solid rgba(255, 255, 255, 0.72);
  border-radius: 8px;
  overflow: hidden;
  color: #ffffff;
  background:
    linear-gradient(120deg, rgba(255, 255, 255, 0.14), transparent 34%, rgba(255, 255, 255, 0.08) 68%, transparent 100%),
    linear-gradient(135deg, #1167ed 0%, #1677ff 42%, #22b7ff 100%);
  background-size: 220% 100%, 100% 100%;
  animation: blueSheen 12s ease-in-out infinite;
  box-shadow: 0 18px 48px rgba(22, 119, 255, 0.2);
  position: relative;
}

.hero-layout::before,
.hero-layout::after {
  content: "";
  position: absolute;
  pointer-events: none;
}

.hero-layout::before {
  width: 360px;
  height: 120px;
  right: -96px;
  top: 18px;
  opacity: 0.22;
  background:
    linear-gradient(90deg, rgba(255, 255, 255, 0.82) 0 2px, transparent 2px 22px),
    linear-gradient(0deg, rgba(255, 255, 255, 0.64) 0 2px, transparent 2px 22px);
  background-size: 22px 22px;
  transform: rotate(-10deg);
  animation: blueprintMove 16s linear infinite;
}

.hero-layout::after {
  width: 190px;
  height: 190px;
  right: 58px;
  top: -76px;
  border-radius: 50%;
  opacity: 0.18;
  background:
    radial-gradient(circle, transparent 46%, rgba(255, 255, 255, 0.95) 47% 49%, transparent 50% 100%),
    conic-gradient(from 0deg, rgba(255,255,255,0), rgba(255,255,255,0.9), rgba(255,255,255,0));
  animation: rotateSoft 22s linear infinite;
}

.hero-main {
  min-width: 0;
  position: relative;
  z-index: 1;
}

.hero-main h1 {
  position: relative;
  margin: 0 0 6px;
  font-size: 30px;
  line-height: 1.18;
  letter-spacing: 0;
}

.ocean-badges {
  position: relative;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 16px;
}

.ocean-badges span,
.panel-hint span {
  display: inline-flex;
  align-items: center;
  min-height: 30px;
  padding: 6px 12px;
  border-radius: 6px;
  color: #0f5fcf;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(215, 231, 247, 0.9);
  font-size: 13px;
  box-shadow: 0 8px 18px rgba(22, 119, 255, 0.08);
}

.hero-nav {
  position: relative;
  z-index: 1;
  align-self: center;
}

.hero-nav button {
  min-width: 148px;
  min-height: 46px;
  border: 0 !important;
  border-radius: 999px !important;
  color: #0f5fcf !important;
  background: rgba(255, 255, 255, 0.95) !important;
  box-shadow: 0 14px 32px rgba(8, 64, 152, 0.22) !important;
}

.panel-hint {
  position: relative;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 14px;
}

.auth-section {
  position: relative;
  margin-bottom: 18px;
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
  box-shadow: 0 10px 24px rgba(34, 99, 174, 0.06);
}

.auth-section > .block,
.auth-section > div {
  overflow: visible !important;
}

.auth-section.user-section {
  min-height: 0;
  z-index: 40;
}

.auth-section.config-section {
  z-index: 20;
}

.auth-section-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 0 10px;
  color: #12243f;
  font-size: 18px;
  font-weight: 700;
}

.auth-section-title::before {
  content: "";
  width: 6px;
  height: 20px;
  border-radius: 999px;
  background: linear-gradient(180deg, #1677ff, #29a8ff);
  box-shadow: 0 8px 18px rgba(22, 119, 255, 0.22);
}

.auth-shell,
.chat-shell {
  position: relative;
  overflow: visible;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px;
  background: var(--panel);
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
}

.auth-shell .form,
.auth-shell .block,
.auth-shell .gap {
  overflow: visible !important;
}

.auth-shell {
  z-index: 30;
}

.chat-shell {
  z-index: 10;
}

.auth-shell::before,
.chat-shell::before {
  content: "";
  position: absolute;
  inset: 0 0 auto 0;
  height: 3px;
  pointer-events: none;
  border-radius: 8px 8px 0 0;
  background: linear-gradient(90deg, #1677ff, #29a8ff);
}

.auth-grid,
.action-row,
.tool-row {
  gap: 10px;
}

.auth-grid {
  align-items: start;
  overflow: visible !important;
  row-gap: 8px !important;
  margin-top: 0 !important;
}

.auth-grid .wrap,
.auth-grid .container,
.auth-grid .form,
.auth-grid .block {
  margin-top: 0 !important;
}

.auth-grid > *,
.tool-row > *,
.action-row > * {
  min-width: 0;
}

.user-row {
  align-items: end;
}

.chatbot-panel {
  border: 1px solid var(--line) !important;
  border-radius: 8px !important;
  overflow: hidden;
  background: linear-gradient(180deg, #ffffff, #f7fbff) !important;
}

.chatbot-panel .message {
  border-radius: 8px !important;
  border: 1px solid rgba(22, 119, 255, 0.12) !important;
  box-shadow: 0 8px 20px rgba(31, 86, 144, 0.08);
}

.chatbot-panel .message.user {
  background: linear-gradient(135deg, #1677ff, #29a8ff) !important;
  color: #ffffff !important;
}

.chatbot-panel .message.bot,
.chatbot-panel .message.assistant {
  background: rgba(255, 255, 255, 0.98) !important;
  color: var(--ink) !important;
}

.message-input textarea {
  min-height: 78px !important;
}

.status-box textarea,
.detail-panel textarea {
  font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace !important;
}

.gradio-container button.primary,
.gradio-container button[variant="primary"] {
  border: 0 !important;
  color: #ffffff !important;
  background: linear-gradient(135deg, #1167ed, #1677ff 55%, #29a8ff) !important;
  box-shadow: 0 10px 24px rgba(22, 119, 255, 0.26) !important;
}

.gradio-container button {
  border-radius: 8px !important;
  border-color: var(--line) !important;
  color: var(--ink) !important;
  background: #ffffff !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.gradio-container button:hover {
  transform: translateY(-1px);
  border-color: rgba(22, 119, 255, 0.36) !important;
  box-shadow: 0 12px 24px rgba(22, 119, 255, 0.14) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .wrap,
.gradio-container .container {
  border-color: var(--line) !important;
  border-radius: 8px !important;
}

.gradio-container input:focus,
.gradio-container textarea:focus {
  border-color: rgba(22, 119, 255, 0.56) !important;
  box-shadow: 0 0 0 3px rgba(22, 119, 255, 0.1) !important;
}

.gradio-container label,
.gradio-container .label-wrap span {
  color: var(--muted) !important;
}

.auth-shell [role="listbox"],
.auth-shell .dropdown-options,
.auth-shell .options,
.auth-shell .options ul {
  z-index: 9999 !important;
}

.auth-shell,
.auth-section,
.auth-grid,
.auth-shell .wrap:has([role="listbox"]) {
  overflow: visible !important;
}

.auth-user-dropdown,
.auth-user-dropdown > *,
.auth-user-dropdown .wrap,
.auth-user-dropdown .container,
.auth-user-dropdown .form,
.auth-user-dropdown .block {
  margin: 0 !important;
}

.report-panel {
  border-radius: 8px;
  overflow: hidden;
}

.resource-actions {
  margin-bottom: 14px;
  gap: 10px;
}

.resource-panel,
.session-summary-panel,
.session-report-panel {
  min-height: 0;
}

.inline-memory-panel {
  margin-bottom: 14px;
}

.fold-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
}

.fold-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: linear-gradient(180deg, #ffffff, #f7fbff);
  box-shadow: 0 12px 26px rgba(34, 99, 174, 0.08);
  overflow: hidden;
}

.fold-card summary {
  cursor: pointer;
  list-style: none;
  padding: 14px 16px;
  color: #10284c;
  font-size: 16px;
  font-weight: 760;
  border-bottom: 1px solid rgba(215, 231, 247, 0.78);
  background: rgba(255, 255, 255, 0.88);
}

.fold-card summary::-webkit-details-marker {
  display: none;
}

.fold-card summary::before {
  content: "+";
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  margin-right: 9px;
  border-radius: 999px;
  color: #ffffff;
  background: linear-gradient(135deg, #1677ff, #29a8ff);
  font-weight: 800;
}

.fold-card[open] summary::before {
  content: "-";
}

.fold-card-body {
  padding: 14px;
}

.fold-card .doc-card {
  margin-bottom: 0;
  box-shadow: none;
}

.consult-shell {
  margin-top: 0;
}

.consult-shell,
.consult-panel {
  background:
    linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246, 251, 255, 0.98));
}

.consult-shell {
  border-color: rgba(22, 119, 255, 0.18);
  box-shadow: 0 18px 42px rgba(32, 96, 170, 0.12), inset 0 1px 0 rgba(255,255,255,.7);
}

.consult-shell::before {
  background: linear-gradient(90deg, #1167ed, #29a8ff 60%, rgba(41,168,255,.55));
}

.hub-actions button {
  min-height: 40px;
  min-width: 132px;
  border-radius: 999px !important;
}

.history-session-select {
  margin-bottom: 12px;
}

.consult-panel .doc-card {
  background: linear-gradient(180deg, #ffffff, #f7fbff);
  border-color: rgba(215, 231, 247, 0.92);
}

.consult-panel .doc-title {
  color: #0f3e8a;
}

.consult-panel .doc-title::before {
  background: linear-gradient(180deg, #1677ff, #7bd8ff);
}

.consult-panel .pretty-list li > span,
.consult-panel .list-index {
  background: linear-gradient(135deg, #1677ff, #29a8ff);
}

.consult-panel .field-key {
  color: #1456c9;
}

.consult-panel .empty {
  background: #f8fcff;
}

.doc-card {
  margin: 0 0 14px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: linear-gradient(180deg, #ffffff, #f7fbff);
  box-shadow: 0 12px 26px rgba(34, 99, 174, 0.08);
}

.doc-title {
  display: flex;
  align-items: center;
  gap: 10px;
  color: #10284c;
  font-size: 18px;
  font-weight: 750;
  margin-bottom: 8px;
}

.doc-title::before {
  content: "";
  width: 8px;
  height: 22px;
  border-radius: 999px;
  background: linear-gradient(180deg, #1677ff, #29a8ff);
}

.doc-subtitle {
  color: var(--muted);
  margin-bottom: 12px;
  font-size: 13px;
}

.doc-body {
  display: grid;
  gap: 10px;
}

.json-row {
  display: grid;
  grid-template-columns: 160px minmax(0, 1fr);
  gap: 12px;
  padding: 12px;
  border: 1px solid rgba(215, 231, 247, 0.78);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.78);
}

.json-key {
  color: #1456c9;
  font-weight: 700;
  line-height: 1.6;
}

.json-value,
.json-text {
  color: #1b2d45;
  line-height: 1.7;
  word-break: break-word;
}

.json-list {
  margin: 0;
  padding-left: 0;
  list-style: none;
}

.json-list li {
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr);
  gap: 10px;
  margin: 8px 0;
  line-height: 1.7;
}

.list-index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 999px;
  color: #ffffff;
  background: #1677ff;
  font-size: 12px;
  font-weight: 700;
}

.list-content {
  min-width: 0;
}

.empty-card {
  padding: 14px;
  color: var(--muted);
  border: 1px dashed var(--line);
  border-radius: 8px;
  background: #fbfdff;
}

.doc-path {
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid var(--line);
  color: #7a8da6;
  font-size: 12px;
  word-break: break-all;
}

.two-report-cols {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.session-report-card .doc-card {
  box-shadow: none;
}

.full-report-shell {
  display: grid;
  gap: 14px;
}

@keyframes workspaceGrid {
  0% { background-position: 0 0, 0 0, 0 0; }
  100% { background-position: 32px 32px, 32px 32px, 0 0; }
}

@keyframes blueSheen {
  0%, 100% { background-position: 0% 0, 0 0; }
  50% { background-position: 100% 0, 0 0; }
}

@keyframes blueprintMove {
  0% { background-position: 0 0, 0 0; }
  100% { background-position: 22px 22px, -22px 22px; }
}

@keyframes rotateSoft {
  to { transform: rotate(360deg); }
}

@media (max-width: 720px) {
  .hero-layout {
    padding: 20px 18px;
  }

  .hero-main h1 {
    font-size: 24px;
  }

  .auth-shell,
  .chat-shell {
    padding: 12px;
  }

  .auth-section {
    padding: 14px;
  }

  .json-row {
    grid-template-columns: 1fr;
  }

  .two-report-cols {
    grid-template-columns: 1fr;
  }

  .fold-grid {
    grid-template-columns: 1fr;
  }

  .hero-layout {
    grid-template-columns: 1fr;
  }

  .hero-nav button {
    width: 100%;
  }
}
"""


def build_app() -> gr.Blocks:
    migrate_legacy_session_tree()
    ensure_seed_users()
    with gr.Blocks(title="MemAgent Chat") as demo:
        state = gr.State(empty_state())

        with gr.Row(elem_classes=["app-shell", "hero-layout"]):
            gr.HTML(
                """
                <div class="hero-main">
                  <h1>MemAgent 咨询系统</h1>
                  <div class="ocean-badges">
                    <span>连续咨询</span>
                    <span>长期记忆</span>
                    <span>阶段摘要</span>
                    <span>报告追踪</span>
                  </div>
                </div>
                """
            )
            with gr.Column(elem_classes=["hero-nav"]):
                consult_nav_btn = gr.Button("我的咨询")

        with gr.Column(visible=True, elem_classes=["auth-shell"]) as auth_page:
            with gr.Group(elem_classes=["auth-section", "user-section"]):
                gr.HTML('<div class="auth-section-title">选择用户</div>')
                with gr.Row(elem_classes=["auth-grid"]):
                    existing_user = gr.Dropdown(
                        choices=collect_known_user_ids(),
                        label="已有用户",
                        allow_custom_value=False,
                        filterable=False,
                        elem_classes=["auth-user-dropdown"],
                    )
                    typed_user = gr.Textbox(label="输入名称 / 用户 ID", interactive=True)
            with gr.Group(elem_classes=["auth-section", "config-section"]):
                gr.HTML('<div class="auth-section-title">配置模型</div>')
                with gr.Row(elem_classes=["auth-grid"]):
                    login_agent_backend = gr.Dropdown(AGENT_BACKEND_CHOICES, value="gpt", label="Empathy Agent")
                    login_summary_backend = gr.Dropdown(SUMMARY_BACKEND_CHOICES, value="gpt", label="Summary Agent")
                login_supervisor_backend = gr.State("gpt")
                login_graph_backend = gr.State("gpt")
            with gr.Row(elem_classes=["action-row"]):
                login_btn = gr.Button("登录", variant="primary")
            auth_status = gr.Textbox(label="认证状态", interactive=False, lines=3, elem_classes=["status-box"])

        with gr.Column(visible=False, elem_classes=["chat-shell"]) as chat_page:
            with gr.Row(elem_classes=["user-row"]):
                current_user = gr.Textbox(label="当前用户", interactive=False)
                logout_btn = gr.Button("退出登录")
            resource_panel = gr.HTML(label="资料展示", elem_classes=["resource-panel", "inline-memory-panel"])
            chatbot = gr.Chatbot(
                label="咨询对话",
                height=520,
                buttons=["copy", "copy_all"],
                layout="bubble",
                elem_classes=["chatbot-panel"],
            )
            user_message = gr.Textbox(
                label="输入消息",
                placeholder="写下此刻想聊的事...",
                lines=3,
                elem_classes=["message-input"],
            )
            with gr.Row(elem_classes=["action-row"]):
                send_btn = gr.Button("发送", variant="primary")
                end_btn = gr.Button("结束当前会话")
            with gr.Row(elem_classes=["session-result-row"]):
                session_summary_panel = gr.HTML(label="本次会话总结", elem_classes=["session-summary-panel"])
                session_report_panel = gr.HTML(label="本次督导师报告", elem_classes=["session-report-panel"])
            chat_status = gr.Textbox(label="会话状态", interactive=False, lines=4, elem_classes=["status-box"])

        with gr.Column(visible=False, elem_classes=["chat-shell", "consult-shell"]) as consult_page:
            with gr.Row(elem_classes=["resource-actions", "hub-actions"]):
                hub_report_btn = gr.Button("完整报告")
                hub_history_btn = gr.Button("历史会话")
                chat_nav_btn = gr.Button("返回会话")
            history_session = gr.Dropdown(
                choices=[],
                label="会话编号",
                allow_custom_value=False,
                filterable=False,
                visible=False,
                elem_classes=["history-session-select"],
            )
            consult_panel = gr.HTML(label="我的咨询", elem_classes=["resource-panel", "consult-panel"])
            consult_status = gr.Textbox(label="咨询资料状态", interactive=False, lines=2, elem_classes=["status-box"])

        login_btn.click(
            fn=login_user,
            inputs=[existing_user, typed_user, login_agent_backend, login_summary_backend, login_supervisor_backend, login_graph_backend],
            outputs=[
                state,
                auth_status,
                auth_page,
                chat_page,
                current_user,
                resource_panel,
                history_session,
                consult_panel,
                consult_page,
            ],
            queue=False,
        )
        logout_btn.click(
            fn=logout_user,
            inputs=[],
            outputs=[
                state,
                auth_status,
                current_user,
                user_message,
                auth_page,
                chat_page,
                consult_page,
                resource_panel,
                session_summary_panel,
                session_report_panel,
                chat_status,
                consult_panel,
                history_session,
                consult_status,
            ],
            queue=False,
        )

        send_btn.click(
            fn=send_message,
            inputs=[user_message, state],
            outputs=[user_message, state, chatbot, chat_status],
        )
        user_message.submit(
            fn=send_message,
            inputs=[user_message, state],
            outputs=[user_message, state, chatbot, chat_status],
        )
        end_btn.click(
            fn=end_session,
            inputs=[state],
            outputs=[state, chatbot, chat_status, session_summary_panel, session_report_panel, resource_panel],
        )
        chat_nav_btn.click(
            fn=open_chat_page,
            inputs=[state],
            outputs=[chat_page, consult_page, chat_status],
            queue=False,
        )
        consult_nav_btn.click(
            fn=open_consultation_hub,
            inputs=[state],
            outputs=[chat_page, consult_page, consult_panel, history_session, consult_status],
            queue=False,
        )
        hub_report_btn.click(
            fn=show_hub_full_report,
            inputs=[state],
            outputs=[consult_panel, history_session, consult_status],
        )
        hub_history_btn.click(
            fn=show_hub_history,
            inputs=[state],
            outputs=[state, consult_panel, history_session, consult_status],
        )
        history_session.change(
            fn=select_history_session,
            inputs=[history_session, state],
            outputs=[state, consult_panel, consult_status],
        )
        demo.load(fn=warmup_ui_callback, inputs=[], outputs=[], queue=False)

    return demo


def find_available_port(host: str, preferred_port: int, attempts: int = 20) -> int:
    for port in range(preferred_port, preferred_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) != 0:
                return port
    raise OSError(f"Cannot find empty port in range: {preferred_port}-{preferred_port + attempts - 1}.")


def launch_app() -> None:
    demo = build_app()
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(
        server_name=server_name,
        server_port=find_available_port(server_name, preferred_port),
        inbrowser=False,
        css=APP_CSS,
    )
