"""
Graph extraction backed by the shared streaming OpenAI-compatible client.

This implementation returns Mem0-ready graph entities and relations without
using non-streaming SDK calls.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv

from multi_agent.utils.openai_client import OpenAIChatClient


logger = logging.getLogger(__name__)


class GraphExtractor:
    """Extract a counseling knowledge graph from dialogue text."""

    def __init__(self) -> None:
        load_dotenv()
        self.provider = os.getenv("GRAPH_LLM_PROVIDER", "openai")
        self.api_key = os.getenv("GRAPH_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("GRAPH_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("GRAPH_LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.4")
        self.client = None

        if not self.api_key or not self.base_url:
            logger.warning(
                "GRAPH_LLM credentials are not configured; graph extraction will return an empty graph"
            )
            return

        self.client = OpenAIChatClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )

    def extract_graph(self, conversation_text: str, user_id: str = "user") -> Dict[str, Any]:
        if self.client is None:
            return {"entities": [], "relations": []}

        try:
            response_text, _usage = self.client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract a counseling knowledge graph from dialogue. "
                            "Return valid JSON only with top-level keys 'entities' and 'relations'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"{self._build_instruction(user_id)}\n\n## 对话历史\n{conversation_text}",
                    },
                ]
            )
            graph_data = self._parse_graph_json(response_text)
            graph_data.setdefault("entities", [])
            graph_data.setdefault("relations", [])
            if not isinstance(graph_data["entities"], list):
                graph_data["entities"] = []
            if not isinstance(graph_data["relations"], list):
                graph_data["relations"] = []
            return graph_data
        except Exception as exc:
            logger.error("Graph extraction failed: %s", exc)
            return {"entities": [], "relations": []}

    def _parse_graph_json(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"entities": [], "relations": []}

        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            return {"entities": [], "relations": []}

        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError:
            logger.warning("Graph extractor returned non-JSON content: %r", text[:200])
            return {"entities": [], "relations": []}

        if not isinstance(data, dict):
            return {"entities": [], "relations": []}
        return data

    def _build_instruction(self, user_id: str) -> str:
        return f"""你是一个心理咨询知识图谱抽取助手。请从来访与咨询师的对话中，抽取可用于记忆检索的实体和关系。

实体类型只允许使用以下类别：
- user: 来访者，固定使用 {user_id}
- assistant: 咨询师，固定使用 assistant
- symptom: 症状，如失眠、头痛、心慌、食欲下降
- emotion: 情绪，如焦虑、无力、委屈、自责、希望
- cognition: 自动化想法或核心信念
- behavior: 行为或应对方式
- life_event: 生活事件、学习事件、家庭事件
- interest: 兴趣、偏好、价值取向
- goal: 计划、目标、探索方向
- strategy: 咨询策略、作业或应对策略

关系类型只允许使用以下类别：
- reports
- feels
- believes
- does
- experiences
- interested_in
- wants
- conflicts_with
- causes
- relieves
- applies
- supports

输出要求：
- 严格输出 JSON
- 只能包含两个顶层字段：entities, relations
- entities 形如 {{"entity": "名称", "entity_type": "类型"}}
- relations 形如 {{"source": "实体", "relation": "关系", "destination": "实体"}}
- 只抽取对话中明确出现或高度可支持的信息
- 不要输出解释、注释、Markdown

{{
  "entities": [{{"entity": "实体名称", "entity_type": "实体类型"}}],
  "relations": [{{"source": "源实体", "relation": "关系类型", "destination": "目标实体"}}]
}}
"""
