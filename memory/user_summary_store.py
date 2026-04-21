import json
from pathlib import Path
from typing import Any, Dict, Optional


class UserSummaryStore:
    def __init__(self, storage_dir: str = "./multi_agent_user_summaries"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_user_summary(self, user_id: str, summary: Dict[str, Any]) -> None:
        path = self.storage_dir / f"{user_id}_summary.json"
        validated = {
            "主题": summary.get("主题", "未知"),
            "背景": summary.get("背景", "用户信息不足"),
            "会话总结": summary.get("会话总结", "暂无对话历史"),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(validated, handle, ensure_ascii=False, indent=2)

    def get_user_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        path = self.storage_dir / f"{user_id}_summary.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None
