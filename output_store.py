from __future__ import annotations

import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "test_outputs"
CONSULTING_REPORTS_DIR = OUTPUT_ROOT / "consulting_reports"
SESSIONS_DIR = OUTPUT_ROOT / "sessions"
SESSION_SUMMARIES_DIR = OUTPUT_ROOT / "sessions_summaries"
TRENDS_DIR = OUTPUT_ROOT / "trends"
WHOLE_SUMMARIES_DIR = OUTPUT_ROOT / "whole_summaries"
USER_REGISTRY_PATH = OUTPUT_ROOT / "users.json"


def ensure_output_dirs() -> None:
    for directory in [
        OUTPUT_ROOT,
        CONSULTING_REPORTS_DIR,
        SESSIONS_DIR,
        SESSION_SUMMARIES_DIR,
        TRENDS_DIR,
        WHOLE_SUMMARIES_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def safe_user_id(user_id: str) -> str:
    text = str(user_id or "").strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown_user"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sessions_path(user_id: str) -> Path:
    return SESSIONS_DIR / f"{safe_user_id(user_id)}.json"


def sessions_summaries_path(user_id: str) -> Path:
    return SESSION_SUMMARIES_DIR / f"{safe_user_id(user_id)}.json"


def session_summaries_path(user_id: str) -> Path:
    return sessions_summaries_path(user_id)


def trends_path(user_id: str) -> Path:
    return TRENDS_DIR / f"{safe_user_id(user_id)}.json"


def whole_summary_path(user_id: str) -> Path:
    return WHOLE_SUMMARIES_DIR / f"{safe_user_id(user_id)}_summary.json"


def consulting_report_json_path(user_id: str, prefix: str = "report") -> Path:
    return CONSULTING_REPORTS_DIR / f"{safe_user_id(user_id)}_{prefix}.json"


def consulting_report_html_path(user_id: str, prefix: str = "report") -> Path:
    return CONSULTING_REPORTS_DIR / f"{safe_user_id(user_id)}_{prefix}.html"


def consulting_report_log_path(user_id: str) -> Path:
    return CONSULTING_REPORTS_DIR / f"{safe_user_id(user_id)}_runtime_trace.log"


def consulting_report_checkpoint_path(user_id: str) -> Path:
    return CONSULTING_REPORTS_DIR / f"{safe_user_id(user_id)}_replay_checkpoint.json"


def _session_index_from_label(session_id: Optional[str]) -> Optional[int]:
    if not session_id:
        return None
    match = re.fullmatch(r"Session(\d+)", str(session_id).strip())
    if not match:
        return None
    return int(match.group(1))


def _next_session_index(sessions: List[Dict[str, Any]]) -> int:
    indexes = []
    for item in sessions:
        if isinstance(item.get("id"), int):
            indexes.append(item["id"])
        label_index = _session_index_from_label(item.get("session_id"))
        if label_index is not None:
            indexes.append(label_index)
    return (max(indexes) + 1) if indexes else 1


def _find_existing_session_identity(
    items: List[Dict[str, Any]],
    *,
    run_session_id: Optional[str] = None,
    session_id: Optional[str] = None,
    content_fingerprint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if run_session_id:
        for item in items:
            if item.get("run_session_id") == run_session_id:
                return item
    if session_id:
        for item in items:
            if item.get("session_id") == session_id:
                return item
    if content_fingerprint:
        for item in items:
            if item.get("content_fingerprint") == content_fingerprint:
                return item
    return None


def _resolve_session_identity(
    *,
    run_session_id: str,
    logical_session_id: Optional[str],
    content_fingerprint: Optional[str],
    existing_items: List[Dict[str, Any]],
) -> tuple[int, str]:
    existing_by_content = _find_existing_session_identity(
        existing_items,
        content_fingerprint=content_fingerprint,
    )
    if existing_by_content:
        existing_label = existing_by_content.get("session_id")
        session_index = (
            existing_by_content.get("id") if isinstance(existing_by_content.get("id"), int) else None
        ) or _session_index_from_label(existing_label) or _next_session_index(existing_items)
        return session_index, existing_label or f"Session{session_index}"

    if logical_session_id:
        session_id = logical_session_id
        existing = _find_existing_session_identity(
            existing_items,
            run_session_id=run_session_id,
            session_id=session_id,
        )
        session_index = (
            _session_index_from_label(session_id)
            or (existing.get("id") if existing and isinstance(existing.get("id"), int) else None)
            or _next_session_index(existing_items)
        )
        return session_index, session_id

    existing = _find_existing_session_identity(existing_items, run_session_id=run_session_id)
    if existing:
        existing_label = existing.get("session_id")
        session_index = (
            existing.get("id") if isinstance(existing.get("id"), int) else None
        ) or _session_index_from_label(existing_label) or _next_session_index(existing_items)
        return session_index, existing_label or f"Session{session_index}"

    session_index = _next_session_index(existing_items)
    return session_index, f"Session{session_index}"


def _split_outputs_contain_run(user_id: str, run_session_id: str) -> bool:
    sessions_doc = read_json(sessions_path(user_id), {"sessions": []})
    summaries_doc = read_json(sessions_summaries_path(user_id), {"sessions": []})
    trends_doc = read_json(trends_path(user_id), {"reports": []})
    return all(
        _find_existing_session_identity(items, run_session_id=run_session_id)
        for items in [
            sessions_doc.get("sessions", []),
            summaries_doc.get("sessions", []),
            trends_doc.get("reports", []),
        ]
    )


def _turns_from_chat_pairs(chat_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for index, pair in enumerate(chat_pairs, start=1):
        turns.append(
            {
                "turn_id": f"Turn{index}",
                "user": pair.get("user", ""),
                "assistant": pair.get("assistant", ""),
            }
        )
    return turns


def _normalized_chat_pairs(chat_pairs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "user": str(pair.get("user", "")).strip(),
            "assistant": str(pair.get("assistant", "")).strip(),
        }
        for pair in chat_pairs
    ]


def _content_fingerprint(chat_pairs: List[Dict[str, Any]]) -> str:
    normalized = _normalized_chat_pairs(chat_pairs)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _upsert_by_session_id(items: List[Dict[str, Any]], entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    entry_session_id = entry.get("session_id")
    entry_run_session_id = entry.get("run_session_id")
    entry_fingerprint = entry.get("content_fingerprint")
    for index, item in enumerate(items):
        if (entry_session_id and item.get("session_id") == entry_session_id) or (
            entry_run_session_id and item.get("run_session_id") == entry_run_session_id
        ) or (
            entry_fingerprint and item.get("content_fingerprint") == entry_fingerprint
        ):
            entry.setdefault("id", item.get("id"))
            entry.setdefault("session_id", item.get("session_id"))
            items[index] = entry
            return items
    items.append(entry)
    return items


def record_session_outputs(
    *,
    user_id: str,
    display_name: str,
    run_session_id: str,
    chat_pairs: List[Dict[str, Any]],
    l2_summary: Optional[Dict[str, Any]],
    treatment_report: Optional[Dict[str, Any]],
    l1_summary: Optional[Dict[str, Any]],
    archived_at: Optional[str] = None,
    logical_session_id: Optional[str] = None,
) -> Dict[str, str]:
    ensure_output_dirs()
    archived_at = archived_at or datetime.now().isoformat()

    sessions_doc = read_json(
        sessions_path(user_id),
        {"user_id": user_id, "display_name": display_name, "updated_at": "", "sessions": []},
    )
    sessions = sessions_doc.setdefault("sessions", [])
    summary_doc = read_json(
        sessions_summaries_path(user_id),
        {"user_id": user_id, "display_name": display_name, "updated_at": "", "sessions": []},
    )
    summaries = summary_doc.setdefault("sessions", [])
    trends_doc = read_json(
        trends_path(user_id),
        {"user_id": user_id, "display_name": display_name, "updated_at": "", "reports": []},
    )
    reports = trends_doc.setdefault("reports", [])
    content_fingerprint = _content_fingerprint(chat_pairs)
    session_index, session_id = _resolve_session_identity(
        run_session_id=run_session_id,
        logical_session_id=logical_session_id,
        content_fingerprint=content_fingerprint,
        existing_items=sessions + summaries + reports,
    )

    session_entry = {
        "id": session_index,
        "session_id": session_id,
        "run_session_id": run_session_id,
        "content_fingerprint": content_fingerprint,
        "archived_at": archived_at,
        "turn_count": len(chat_pairs),
        "turns": _turns_from_chat_pairs(chat_pairs),
    }
    sessions_doc.update({"user_id": user_id, "display_name": display_name, "updated_at": archived_at})
    sessions_doc["sessions"] = sorted(
        _upsert_by_session_id(sessions, session_entry),
        key=lambda item: item.get("id", 0),
    )
    write_json(sessions_path(user_id), sessions_doc)

    summary_entry = {
        "id": session_index,
        "session_id": session_id,
        "run_session_id": run_session_id,
        "content_fingerprint": content_fingerprint,
        "archived_at": archived_at,
        "summary": l2_summary or {},
    }
    summary_doc.update({"user_id": user_id, "display_name": display_name, "updated_at": archived_at})
    summary_doc.pop("session_summaries", None)
    summary_doc["sessions"] = sorted(
        _upsert_by_session_id(summaries, summary_entry),
        key=lambda item: item.get("id", 0),
    )
    write_json(sessions_summaries_path(user_id), summary_doc)

    trend_entry = {
        "id": session_index,
        "session_id": session_id,
        "run_session_id": run_session_id,
        "content_fingerprint": content_fingerprint,
        "generated_at": archived_at,
        "report": treatment_report or {},
    }
    trends_doc.update({"user_id": user_id, "display_name": display_name, "updated_at": archived_at})
    trends_doc["reports"] = sorted(
        _upsert_by_session_id(reports, trend_entry),
        key=lambda item: item.get("id", 0),
    )
    write_json(trends_path(user_id), trends_doc)

    if l1_summary is not None:
        write_json(whole_summary_path(user_id), l1_summary)

    return {
        "session_id": session_id,
        "sessions": str(sessions_path(user_id).resolve()),
        "sessions_summaries": str(sessions_summaries_path(user_id).resolve()),
        "trends": str(trends_path(user_id).resolve()),
        "whole_summary": str(whole_summary_path(user_id).resolve()),
    }


def load_user_sessions(user_id: str) -> Dict[str, Any]:
    return read_json(sessions_path(user_id), {"user_id": user_id, "sessions": []})


def load_user_session_summaries(user_id: str) -> Dict[str, Any]:
    return read_json(sessions_summaries_path(user_id), {"user_id": user_id, "sessions": []})


def load_user_trends(user_id: str) -> Dict[str, Any]:
    return read_json(trends_path(user_id), {"user_id": user_id, "reports": []})


def migrate_legacy_session_file(path: Path) -> Optional[Dict[str, str]]:
    payload = read_json(path, None)
    if not isinstance(payload, dict):
        return None
    user_id = payload.get("user_id")
    if not user_id:
        return None
    run_session_id = str(payload.get("session_id") or path.stem)

    if _split_outputs_contain_run(str(user_id), run_session_id):
        return {"skipped_legacy_session": str(path.resolve()), "reason": "already_migrated"}

    output_paths = record_session_outputs(
        user_id=str(user_id),
        display_name=str(payload.get("display_name") or user_id),
        run_session_id=run_session_id,
        chat_pairs=payload.get("chat_pairs", []) or [],
        l2_summary=payload.get("l2_summary") or payload.get("session_summary") or {},
        treatment_report=payload.get("treatment_report") or {},
        l1_summary=payload.get("l1_summary"),
        archived_at=payload.get("archived_at") or datetime.now().isoformat(),
    )

    output_paths["migrated_legacy_session"] = str(path.resolve())
    return output_paths


def migrate_legacy_session_tree() -> List[Dict[str, str]]:
    ensure_output_dirs()
    return []
