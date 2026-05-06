from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def print_block(title: str, payload=None) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if payload is not None:
        if isinstance(payload, (dict, list)):
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload)


def main() -> None:
    load_dotenv()

    from agents.empathy_agent import EmpathyAgent

    user_id = os.getenv("MA_USER_ID", "CP224_刘某")
    user_text = os.getenv("L3_TEST_MESSAGE", "我最近总是失眠，而且长时间感觉大脑神经紧绷却又休息浅，时间久了感到很疲惫，已经影响到了精神状态和日常生活的效率。但是睡眠少还是会很早醒来然后睡不着回笼觉，处于无效睡眠状态。试过用身体的劳累来增加睡眠的深度，但好像效果不太明显。而且睡眠不足导致运动时也会担心身体状态，感觉自己处于一个恶性循环中。")

    print_block(
        "CONFIG",
        {
            "user_id": user_id,
            "message": user_text,
            "agent_backend": os.getenv("MA_AGENT_MODEL_BACKEND", "gpt"),
            "summary_backend": os.getenv("MA_SUMMARY_MODEL_BACKEND", "gpt"),
            "supervisor_backend": os.getenv("MA_SUPERVISOR_MODEL_BACKEND", "gpt"),
            "graph_backend": os.getenv("MA_GRAPH_MODEL_BACKEND", "gpt"),
        },
    )

    try:
        agent = EmpathyAgent(
            model_backend=os.getenv("MA_AGENT_MODEL_BACKEND", "gpt"),
            summary_model_backend=os.getenv("MA_SUMMARY_MODEL_BACKEND", "gpt"),
            supervisor_model_backend=os.getenv("MA_SUPERVISOR_MODEL_BACKEND", "gpt"),
            graph_model_backend=os.getenv("MA_GRAPH_MODEL_BACKEND", "gpt"),
        )
        session_info = agent.start_session(user_id)
        print_block("SESSION", session_info)
    except Exception:
        print_block("INIT FAILED")
        traceback.print_exc()
        return

    try:
        result = agent.generate_response(user_text)
        print_block("MODEL RESPONSE", result.get("response", ""))
        print_block("L3 RESULT", result.get("l3_memory_result", {}))
        print_block("RETRIEVAL CONTEXT", result.get("retrieval_context", ""))
    except Exception:
        print_block("GENERATE FAILED")
        traceback.print_exc()
        return

    try:
        memory_dump = agent.mem0.search_memories(user_id, user_text, limit=10)
        print_block("MEMORY SEARCH AFTER WRITE", memory_dump)
    except Exception:
        print_block("SEARCH FAILED")
        traceback.print_exc()


if __name__ == "__main__":
    main()
