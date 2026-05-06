import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


class UserSummaryStore:
    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir is None:
            self.storage_dir = Path(__file__).resolve().parents[1] / "test_outputs" / "whole_summaries"
        else:
            self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, user_id: str) -> Path:
        safe_user_id = re.sub(r"\s+", "_", str(user_id or "").strip())
        safe_user_id = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", safe_user_id)
        safe_user_id = re.sub(r"_+", "_", safe_user_id).strip("_") or "unknown_user"
        return self.storage_dir / f"{safe_user_id}_summary.json"

    def save_user_summary(self, user_id: str, summary: Dict[str, Any]) -> None:
        path = self._path_for(user_id)
        validated = {
            "主题": summary.get("主题", "未知"),
            "背景": summary.get("背景", "用户信息不足"),
            "会话总结": summary.get("会话总结", "暂无对话历史"),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(validated, handle, ensure_ascii=False, indent=2)

    def get_user_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        path = self._path_for(user_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None
