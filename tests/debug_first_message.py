from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    load_dotenv()

    print_section("ENV")
    keys = [
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "MA_UI_MODE",
        "MA_AGENT_MODEL_BACKEND",
        "MA_SUMMARY_MODEL_BACKEND",
        "MA_SUPERVISOR_MODEL_BACKEND",
        "MA_GRAPH_MODEL_BACKEND",
        "MEM0_L3_LLM_MODEL_BACKEND",
    ]
    for key in keys:
        print(f"{key}={os.getenv(key, '')}")

    print_section("IMPORT")
    try:
        from agents.empathy_agent import EmpathyAgent
        print("EmpathyAgent import ok")
    except Exception:
        traceback.print_exc()
        return

    print_section("AGENT INIT")
    try:
        agent = EmpathyAgent(
            model_backend=os.getenv("MA_AGENT_MODEL_BACKEND", "gpt"),
            model_mode=os.getenv("MA_AGENT_MODEL_MODE"),
            local_model_path=os.getenv("MA_LOCAL_MODEL_PATH"),
            local_base_model_path=os.getenv("MA_LOCAL_BASE_MODEL_PATH"),
            summary_model_backend=os.getenv("MA_SUMMARY_MODEL_BACKEND", "gpt"),
            summary_model_mode=os.getenv("MA_SUMMARY_MODEL_MODE"),
            summary_local_model_path=os.getenv("MA_SUMMARY_LOCAL_MODEL_PATH"),
            summary_local_base_model_path=os.getenv("MA_SUMMARY_LOCAL_BASE_MODEL_PATH"),
            supervisor_model_backend=os.getenv("MA_SUPERVISOR_MODEL_BACKEND", "gpt"),
            supervisor_model_mode=os.getenv("MA_SUPERVISOR_MODEL_MODE"),
            supervisor_local_model_path=os.getenv("MA_SUPERVISOR_LOCAL_MODEL_PATH"),
            supervisor_local_base_model_path=os.getenv("MA_SUPERVISOR_LOCAL_BASE_MODEL_PATH"),
            graph_model_backend=os.getenv("MA_GRAPH_MODEL_BACKEND", "gpt"),
            graph_model_mode=os.getenv("MA_GRAPH_MODEL_MODE"),
            graph_local_model_path=os.getenv("MA_GRAPH_LOCAL_MODEL_PATH"),
            graph_local_base_model_path=os.getenv("MA_GRAPH_LOCAL_BASE_MODEL_PATH"),
        )
        print("agent init ok")
    except Exception:
        traceback.print_exc()
        return

    user_id = os.getenv("MA_USER_ID", "CP224_刘某")

    print_section("START SESSION")
    try:
        session_info = agent.start_session(user_id)
        print(session_info)
    except Exception:
        traceback.print_exc()
        return

    print_section("FIRST MESSAGE")
    try:
        result = agent.generate_response("测试：我今天有点焦虑，想看看系统会在哪里报错。")
        print("response:", result.get("response", ""))
        print("l3_memory_result:", result.get("l3_memory_result", {}))
        print("retrieval_context:", result.get("retrieval_context", ""))
    except Exception:
        traceback.print_exc()
        return

    print_section("DONE")
    print("first message path ok")


if __name__ == "__main__":
    main()
