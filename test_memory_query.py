from multi_agent.agents.summary_agent import SummaryAgent
from multi_agent.memory.mem0_adapter import Mem0Adapter


USER_ID = "test_user_001"
QUERY = "焦虑 睡眠 工作压力"


def main() -> None:
    mem0 = Mem0Adapter()
    summary_agent = SummaryAgent()

    print("===== L1 =====")
    l1 = summary_agent.get_user_summary(USER_ID)
    print(l1)

    print("\n===== L2 =====")
    l2_list = mem0.get_l2_summaries(USER_ID, limit=10)
    for index, item in enumerate(l2_list, 1):
        print(f"[{index}] {item}")

    print("\n===== L2 + L3 + Graph 检索 =====")
    result = mem0.search_memories(USER_ID, QUERY, limit=5)
    print(result)

    print("\n===== 单独展开 L3 =====")
    for index, item in enumerate(result.get("l3_fragments", []), 1):
        print(f"[{index}] {item}")

    print("\n===== 单独展开 Graph =====")
    for index, item in enumerate(result.get("session_graphs", []), 1):
        print(f"[{index}] {item}")


if __name__ == "__main__":
    main()
