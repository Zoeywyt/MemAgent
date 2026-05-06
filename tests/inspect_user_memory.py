from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def print_title(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def normalize_memory_item(item: Dict[str, Any]) -> Dict[str, Any]:
    metadata = item.get("metadata", {}) or {}
    return {
        "id": item.get("id"),
        "memory": item.get("memory", ""),
        "run_id": item.get("run_id"),
        "memory_type": metadata.get("memory_type"),
        "session_id": metadata.get("session_id"),
        "turn_index": metadata.get("turn_index"),
        "timestamp": metadata.get("timestamp"),
        "topic": metadata.get("topic"),
    }


def load_archived_graph_relations(user_id: str) -> List[Dict[str, Any]]:
    try:
        from output_store import load_user_sessions
    except Exception:
        return []

    relations: List[Dict[str, Any]] = []
    payload = load_user_sessions(user_id)
    for session in payload.get("sessions", []) or []:
        for turn in session.get("turns", []) or []:
            relations.append(
                {
                    "archive_file": "test_outputs/sessions",
                    "session_id": session.get("session_id", ""),
                    "turn_id": turn.get("turn_id", ""),
                    "user": turn.get("user", ""),
                    "assistant": turn.get("assistant", ""),
                }
            )
    return relations


def main() -> None:
    load_dotenv()
    user_id = os.getenv("MA_USER_ID", "wyt")

    print_title("USER")
    print(user_id)

    from memory.mem0_adapter import Mem0Adapter

    mem0 = Mem0Adapter()
    data = mem0.client.get_all(user_id=user_id, limit=500)
    results = data.get("results", []) or []

    l3_items = [normalize_memory_item(item) for item in results if (item.get("metadata", {}) or {}).get("memory_type") == "l3_fragment"]
    session_graph_items = [
        normalize_memory_item(item) for item in results if (item.get("metadata", {}) or {}).get("memory_type") == "session_graph"
    ]
    treatment_reports = [
        normalize_memory_item(item) for item in results if (item.get("metadata", {}) or {}).get("memory_type") == "treatment_report"
    ]
    l2_items = [normalize_memory_item(item) for item in results if (item.get("metadata", {}) or {}).get("memory_type") == "l2_summary"]

    print_title("COUNTS")
    print(
        json.dumps(
            {
                "total_memories": len(results),
                "l3_fragments": len(l3_items),
                "session_graph": len(session_graph_items),
                "l2_summary": len(l2_items),
                "treatment_report": len(treatment_reports),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print_title("L3 FRAGMENTS")
    print(json.dumps(l3_items, ensure_ascii=False, indent=2))

    print_title("SESSION GRAPH MEMORIES")
    print(json.dumps(session_graph_items, ensure_ascii=False, indent=2))

    print_title("ARCHIVED GRAPH RELATIONS")
    print(json.dumps(load_archived_graph_relations(user_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
