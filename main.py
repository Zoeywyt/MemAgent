import os

from multi_agent.agents.empathy_agent import EmpathyAgent


EXIT_COMMANDS = {"/quit", "/exit"}
END_COMMANDS = {"/end", "/save"}


def main() -> None:
    user_id = os.getenv("MA_USER_ID", "user_001")
    agent = EmpathyAgent()
    session_info = agent.start_session(user_id)

    print(f"已启动会话，user_id={user_id}")
    print("输入内容后回车即可继续对话。")
    print("输入 /end 保存会话并生成 L1/L2/图记忆/报告。")
    print("输入 /quit 直接退出，不结束当前会话。")
    print(f"当前 L1 摘要：{session_info.get('l1_summary')}")

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
            end_result = agent.end_session()
            print("\n会话已结束，已保存相关记忆。")
            print(end_result)
            break

        try:
            result = agent.generate_response(user_input)
            print(f"\n模型：{result['response']}")
        except Exception as exc:
            print(f"\n调用失败：{exc}")


if __name__ == "__main__":
    main()
