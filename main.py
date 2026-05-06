import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from agents.empathy_agent import EmpathyAgent
from gradio_ui import launch_app
from output_store import record_session_outputs


EXIT_COMMANDS = {"/quit", "/exit"}
END_COMMANDS = {"/end", "/save"}


def build_agent_from_env() -> EmpathyAgent:
    return EmpathyAgent(
        model_backend=os.getenv("MA_AGENT_MODEL_BACKEND"),
        model_mode=os.getenv("MA_AGENT_MODEL_MODE"),
        local_model_path=os.getenv("MA_LOCAL_MODEL_PATH"),
        local_base_model_path=os.getenv("MA_LOCAL_BASE_MODEL_PATH"),
        summary_model_backend=os.getenv("MA_SUMMARY_MODEL_BACKEND"),
        summary_model_mode=os.getenv("MA_SUMMARY_MODEL_MODE"),
        summary_local_model_path=os.getenv("MA_SUMMARY_LOCAL_MODEL_PATH"),
        summary_local_base_model_path=os.getenv("MA_SUMMARY_LOCAL_BASE_MODEL_PATH"),
        supervisor_model_backend=os.getenv("MA_SUPERVISOR_MODEL_BACKEND"),
        supervisor_model_mode=os.getenv("MA_SUPERVISOR_MODEL_MODE"),
        supervisor_local_model_path=os.getenv("MA_SUPERVISOR_LOCAL_MODEL_PATH"),
        supervisor_local_base_model_path=os.getenv("MA_SUPERVISOR_LOCAL_BASE_MODEL_PATH"),
        graph_model_backend=os.getenv("MA_GRAPH_MODEL_BACKEND"),
        graph_model_mode=os.getenv("MA_GRAPH_MODEL_MODE"),
        graph_local_model_path=os.getenv("MA_GRAPH_LOCAL_MODEL_PATH"),
        graph_local_base_model_path=os.getenv("MA_GRAPH_LOCAL_BASE_MODEL_PATH"),
    )


def run_cli_chat() -> None:
    user_id = os.getenv("MA_USER_ID", "user_001")
    agent = build_agent_from_env()
    session_info = agent.start_session(user_id)

    print(f"已启动会话，user_id={user_id}")
    print("输入内容后回车即可继续对话。")
    print("输入 /end 保存会话并生成 L1/L2/图记忆/报告。")
    print("输入 /quit 直接退出，不结束当前会话。")
    print(f"当前 L1 摘要：{session_info.get('l1_summary')}")
    print(f"已预加载历史会话摘要数：{len(session_info.get('l2_summaries') or [])}")
    print(f"已预加载督导师报告：{'是' if session_info.get('treatment_report') else '否'}")

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n检测到退出，未自动结束会话。")
            break

        if not user_input:
            continue

        if user_input in EXIT_COMMANDS:
            print("已退出。当前会话未执行 end_session()。")
            break

        if user_input in END_COMMANDS:
            if not agent.session_messages:
                print("当前会话没有对话内容，无需归档。")
                break
            run_session_id = agent.session_id or session_info.get("session_id", "")
            chat_pairs = []
            for index in range(0, len(agent.session_messages), 2):
                user_msg = agent.session_messages[index] if index < len(agent.session_messages) else {}
                assistant_msg = agent.session_messages[index + 1] if index + 1 < len(agent.session_messages) else {}
                if user_msg.get("role") == "user":
                    chat_pairs.append({"user": user_msg.get("content", ""), "assistant": assistant_msg.get("content", "")})
            end_result = agent.end_session()
            output_paths = record_session_outputs(
                user_id=user_id,
                display_name=user_id,
                run_session_id=run_session_id,
                chat_pairs=chat_pairs,
                l2_summary=end_result.get("l2_summary", {}),
                treatment_report=end_result.get("treatment_report", {}),
                l1_summary=end_result.get("l1_summary", {}),
            )
            print("\n会话已结束，已保存相关记忆。")
            print(end_result)
            print(output_paths)
            break

        try:
            result = agent.generate_response(user_input)
            print(f"\n模型：{result['response']}")
        except Exception as exc:
            print(f"\n调用失败：{exc}")


def main() -> None:
    ui_mode = os.getenv("MA_UI_MODE", "gradio").strip().lower()
    if ui_mode == "cli":
        run_cli_chat()
        return
    launch_app()


if __name__ == "__main__":
    main()
