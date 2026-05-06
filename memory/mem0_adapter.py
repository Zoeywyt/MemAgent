"""
Mem0 adapter for the multi_agent package.

This implementation uses real Mem0 vector retrieval together with the built-in
Kuzu graph backend. A local streaming patch is applied to Mem0's OpenAI LLM so
the graph pipeline can work with gateways that require `stream=true`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterator, List, Optional

import torch
from dotenv import load_dotenv

try:
    from mem0 import Memory
except ImportError:  # pragma: no cover
    Memory = None  # type: ignore[assignment]

from memory.mem0_stream_patch import patch_mem0_openai_streaming
from utils.model_client import ChatClientProtocol, build_chat_client, resolve_model_mode


logger = logging.getLogger(__name__)


L3_FACT_EXTRACTION_PROMPT = """You are a memory extraction assistant for long-term conversation memory.
Your job is to extract durable user facts from a dialogue turn.

Focus on memory-worthy information such as:
- persistent preferences
- personal background and identity details
- relationships and important people
- plans, goals, and intentions
- stable health, emotional, or work-related patterns
- repeated concerns that may matter in future support

Rules:
- extract facts from the user's messages only
- do not store generic chit-chat or one-off filler
- do not copy assistant claims as facts about the user
- if there is nothing worth remembering, return an empty list
- return valid JSON with a top-level "facts" array of strings
"""


L3_UPDATE_DECISION_PROMPT = """You are a memory manager that updates a user's long-term memory.
Compare newly extracted facts with existing memories and decide, for each fact, whether to:
- ADD: new important information
- UPDATE: same memory slot but the new fact is more complete or replaces the old one
- DELETE: the new fact directly contradicts an existing memory
- NONE: no change is needed

Rules:
- keep memories concise and reusable for future retrieval
- prefer the more informative wording when two memories mean the same thing
- only use existing IDs for UPDATE, DELETE, or NONE
- use a fresh temporary id for ADD entries when needed
- return valid JSON only, with a top-level "memory" array

Each item must be:
{
  "id": "...",
  "text": "...",
  "event": "ADD|UPDATE|DELETE|NONE",
  "old_memory": ""
}
"""


class _FallbackGraph:
    def add(self, *args: Any, **kwargs: Any) -> None:
        return None


class _FallbackMemoryClient:
    def __init__(self, reason: str = ""):
        self.reason = reason
        self.enable_graph = False
        self.graph = _FallbackGraph()

    def add(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def delete(self, *args: Any, **kwargs: Any) -> None:
        return None

    def delete_all(self, *args: Any, **kwargs: Any) -> None:
        return None

    def search(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"results": [], "relations": []}

    def get(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def get_all(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"results": []}


class Mem0Adapter:
    _instance: Optional["Mem0Adapter"] = None
    _memory_client: Optional["Memory"] = None

    def __new__(cls) -> "Mem0Adapter":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        load_dotenv()
        if Memory is None:
            raise ImportError(
                "mem0ai is not installed. Please install multi_agent requirements before using Mem0Adapter."
            )

        patch_mem0_openai_streaming()

        self._l3_llm_api_key = os.getenv("MEM0_L3_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._l3_llm_base_url = os.getenv("MEM0_L3_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self._l3_llm_model = os.getenv("MEM0_L3_LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.4")
        self._l3_llm_client: Optional[ChatClientProtocol] = None
        self._l3_llm_mode = resolve_model_mode("MEM0_L3_LLM", default="remote")
        l3_backend = os.getenv("MEM0_L3_LLM_MODEL_BACKEND") or os.getenv("MULTI_AGENT_MODEL_BACKEND")
        if l3_backend or self._l3_llm_mode == "local" or (self._l3_llm_api_key and self._l3_llm_base_url):
            self._l3_llm_client = build_chat_client(
                "MEM0_L3_LLM",
                backend=l3_backend,
                mode=self._l3_llm_mode,
                base_url=self._l3_llm_base_url,
                api_key=self._l3_llm_api_key,
                model=self._l3_llm_model,
                local_base_model_path=os.getenv("MEM0_L3_LLM_LOCAL_BASE_MODEL_PATH")
                or os.getenv("LOCAL_BASE_MODEL_PATH"),
            )
        else:
            logger.warning("L3 memory judge LLM is not configured; L3 memory extraction will be skipped")

        self._mem0_llm_api_key = os.getenv("MEM0_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._mem0_llm_base_url = os.getenv("MEM0_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self._mem0_llm_model = os.getenv("MEM0_LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.4")
        self._graph_db_path = os.getenv("KUZU_DB_PATH", "./mem0_kuzu_db")
        self._graph_threshold = float(os.getenv("MEM0_GRAPH_THRESHOLD", "0.7"))
        self._graph_toggle_lock = RLock()
        self._io_lock = RLock()
        embedding_model, embedding_kwargs = self._resolve_embedding_config()

        config: Dict[str, Any] = {
            "version": "v1.1",
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self._mem0_llm_model,
                    "api_key": self._mem0_llm_api_key,
                    "openai_base_url": self._mem0_llm_base_url,
                    "temperature": 0.1,
                    "max_tokens": 2000,
                },
            },
            "graph_store": {
                "provider": "kuzu",
                "config": {
                    "db": self._graph_db_path,
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": self._mem0_llm_model,
                        "api_key": self._mem0_llm_api_key,
                        "openai_base_url": self._mem0_llm_base_url,
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    },
                },
                "threshold": self._graph_threshold,
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": embedding_model,
                    "model_kwargs": embedding_kwargs,
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": os.getenv("MEM0_COLLECTION_NAME", "mem0_empathy_collection"),
                    "path": os.getenv("CHROMA_DB_PATH", "./chroma_db"),
                },
            },
            "history_db_path": os.getenv("MEM0_HISTORY_DB", "./mem0_history.db"),
        }

        self._memory_client = self._initialize_memory_client(config)
        logger.info("Mem0 adapter initialized with Chroma vector store and Kuzu graph store")

    def _initialize_memory_client(self, config: Dict[str, Any]) -> "Memory":
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                return Memory.from_config(config)
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if "disk i/o error" in message or "readonly database" in message or "journal" in message:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                break
        logger.warning("Mem0 init fallback engaged: %s", last_error)
        return _FallbackMemoryClient(reason=str(last_error))  # type: ignore[return-value]

    def _resolve_embedding_config(self) -> tuple[str, Dict[str, Any]]:
        local_path = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "").strip()
        project_default = Path(__file__).resolve().parents[1] / "models" / "bge-small-zh-v1.5"

        candidate = None
        if local_path:
            candidate = Path(local_path).expanduser()
        elif model_name:
            named_path = Path(model_name).expanduser()
            if named_path.exists():
                candidate = named_path
        elif project_default.exists():
            candidate = project_default

        model_ref = "BAAI/bge-small-zh-v1.5"
        model_kwargs: Dict[str, Any] = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        if candidate and candidate.exists():
            model_ref = str(candidate.resolve())
            model_kwargs["local_files_only"] = True
        elif model_name:
            model_ref = model_name

        return model_ref, model_kwargs

    @property
    def client(self) -> "Memory":
        if self._memory_client is None:
            raise RuntimeError("Mem0 client is not initialized")
        return self._memory_client

    @contextmanager
    def _io_serialized(self) -> Iterator[None]:
        with self._io_lock:
            yield

    @contextmanager
    def _graph_temporarily_disabled(self) -> Iterator[None]:
        with self._graph_toggle_lock:
            previous = getattr(self.client, "enable_graph", False)
            self.client.enable_graph = False
            try:
                yield
            finally:
                self.client.enable_graph = previous

    def _vector_only_add(
        self,
        text: str,
        *,
        user_id: str,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            with self._io_serialized(), self._graph_temporarily_disabled():
                self.client.add(text, user_id=user_id, run_id=run_id, metadata=metadata, infer=False)
        except Exception as exc:
            logger.error("Vector add failed for user %s run %s: %s", user_id, run_id, exc)

    def _vector_only_delete(self, memory_id: str) -> None:
        try:
            with self._io_serialized(), self._graph_temporarily_disabled():
                self.client.delete(memory_id)
        except Exception as exc:
            logger.error("Vector delete failed for memory %s: %s", memory_id, exc)

    def _vector_only_search(self, query: str, *, user_id: str, limit: int) -> Dict[str, Any]:
        try:
            with self._io_serialized(), self._graph_temporarily_disabled():
                return self.client.search(query, user_id=user_id, limit=limit)
        except Exception as exc:
            logger.error("Vector search failed for user %s: %s", user_id, exc)
            return {"results": [], "relations": []}

    def _ingest_session_graph(self, graph_text: str, user_id: str, session_id: str) -> None:
        if not getattr(self.client, "enable_graph", False):
            return
        try:
            with self._io_serialized():
                self.client.graph.add(
                    graph_text,
                    {
                        "user_id": user_id,
                        "run_id": session_id,
                    },
                )
        except Exception as exc:
            logger.error("Failed to ingest session graph into Kuzu for session %s: %s", session_id, exc)

    def add_graph_data(
        self,
        user_id: str,
        session_id: str,
        graph_data: Dict[str, Any],
        source_text: Optional[str] = None,
    ) -> None:
        if not graph_data.get("entities") and not graph_data.get("relations"):
            logger.info("Skip empty graph data for session %s", session_id)
            return

        entities_text = "\n".join(
            f"- {entity['entity']} (类型: {entity['entity_type']})"
            for entity in graph_data.get("entities", [])
        )
        relations_text = "\n".join(
            f"- {relation['source']} --{relation['relation']}--> {relation['destination']}"
            for relation in graph_data.get("relations", [])
        )
        graph_description = (
            f"本次会话提取的实体：\n{entities_text}\n\n"
            f"本次会话提取的关系：\n{relations_text}"
        )

        self._vector_only_add(
            graph_description,
            user_id=user_id,
            run_id=session_id,
            metadata={"memory_type": "session_graph"},
        )
        self._ingest_session_graph(source_text or graph_description, user_id=user_id, session_id=session_id)
        logger.info(
            "Stored graph for session %s: %s entities, %s relations",
            session_id,
            len(graph_data.get("entities", [])),
            len(graph_data.get("relations", [])),
        )

    def _call_l3_json_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if self._l3_llm_client is None:
            return {}

        try:
            response_text, _usage = self._l3_llm_client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response_text[start:end])
            return {}
        except Exception as exc:
            logger.error("L3 memory LLM call failed: %s", exc)
            return {}

    def _extract_l3_facts(self, user_input: str, assistant_response: str) -> List[str]:
        payload = {
            "messages": [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_response},
            ]
        }
        result = self._call_l3_json_llm(
            L3_FACT_EXTRACTION_PROMPT,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        facts = result.get("facts", [])
        if not isinstance(facts, list):
            return []
        cleaned: List[str] = []
        for item in facts:
            if isinstance(item, str):
                text = item.strip()
                if text and text not in cleaned:
                    cleaned.append(text)
        return cleaned

    def _search_existing_l3_memories(self, user_id: str, facts: List[str], limit_per_fact: int = 5) -> List[Dict[str, Any]]:
        candidates: Dict[str, Dict[str, Any]] = {}
        for fact in facts:
            try:
                results = self._vector_only_search(fact, user_id=user_id, limit=limit_per_fact)
            except Exception as exc:
                logger.error("L3 candidate search failed for fact '%s': %s", fact, exc)
                continue

            for item in results.get("results", []):
                memory_id = item.get("id")
                metadata = item.get("metadata", {})
                if not memory_id or metadata.get("memory_type") != "l3_fragment":
                    continue
                current = candidates.get(memory_id)
                score = item.get("score") or 0
                record = {
                    "id": memory_id,
                    "text": item.get("memory", ""),
                    "score": score,
                    "metadata": metadata,
                }
                if current is None or score > current.get("score", 0):
                    candidates[memory_id] = record

        ordered = sorted(candidates.values(), key=lambda item: -item.get("score", 0))
        return ordered

    def _decide_l3_actions(self, existing_memories: List[Dict[str, Any]], facts: List[str]) -> List[Dict[str, Any]]:
        if not facts:
            return []

        prompt_payload = {
            "existing_memories": [
                {"id": item.get("id"), "text": item.get("text", "")}
                for item in existing_memories
            ],
            "new_facts": facts,
        }
        result = self._call_l3_json_llm(
            L3_UPDATE_DECISION_PROMPT,
            json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        )
        actions = result.get("memory", [])
        if not isinstance(actions, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            event = str(action.get("event", "")).upper().strip()
            if event not in {"ADD", "UPDATE", "DELETE", "NONE"}:
                continue
            normalized.append(
                {
                    "id": action.get("id"),
                    "text": str(action.get("text", "")).strip(),
                    "event": event,
                    "old_memory": str(action.get("old_memory", "")).strip(),
                }
            )
        return normalized

    def _get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._io_serialized():
                return self.client.get(memory_id)
        except Exception as exc:
            logger.error("Failed to load memory %s: %s", memory_id, exc)
            return None

    def _replace_l3_memory(
        self,
        memory_id: str,
        new_text: str,
        user_id: str,
        session_id: str,
        turn_index: int,
        timestamp: datetime,
    ) -> None:
        existing = self._get_memory_by_id(memory_id)
        metadata = deepcopy(existing.get("metadata", {})) if existing else {}
        metadata.update(
            {
                "memory_type": "l3_fragment",
                "session_id": session_id,
                "turn_index": turn_index,
                "timestamp": timestamp.isoformat(),
            }
        )
        try:
            self._vector_only_delete(memory_id)
        except Exception as exc:
            logger.error("Failed to delete old L3 memory %s: %s", memory_id, exc)
        self._vector_only_add(new_text, user_id=user_id, run_id=session_id, metadata=metadata)

    def remember_l3_turn(
        self,
        user_id: str,
        session_id: str,
        turn_index: int,
        user_input: str,
        assistant_response: str,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        event_time = timestamp or datetime.now()
        facts = self._extract_l3_facts(user_input=user_input, assistant_response=assistant_response)
        if not facts:
            return {"facts": [], "actions": [], "stored": 0}

        existing_memories = self._search_existing_l3_memories(user_id=user_id, facts=facts)
        actions = self._decide_l3_actions(existing_memories=existing_memories, facts=facts)
        if not actions:
            actions = [
                {"id": f"new_{idx}", "text": fact, "event": "ADD", "old_memory": ""}
                for idx, fact in enumerate(facts, start=1)
            ]

        existing_by_id = {item.get("id"): item for item in existing_memories if item.get("id")}
        stored_count = 0
        for action in actions:
            event = action["event"]
            memory_id = action.get("id")
            text = action.get("text", "")

            if event == "NONE":
                continue

            if event == "ADD":
                if not text:
                    continue
                metadata = {
                    "memory_type": "l3_fragment",
                    "session_id": session_id,
                    "turn_index": turn_index,
                    "timestamp": event_time.isoformat(),
                }
                self._vector_only_add(text, user_id=user_id, run_id=session_id, metadata=metadata)
                stored_count += 1
                continue

            if event == "DELETE":
                if not memory_id:
                    continue
                try:
                    self._vector_only_delete(memory_id)
                except Exception as exc:
                    logger.error("Failed to delete L3 memory %s: %s", memory_id, exc)
                continue

            if event == "UPDATE":
                if not memory_id or not text:
                    continue
                if memory_id not in existing_by_id:
                    metadata = {
                        "memory_type": "l3_fragment",
                        "session_id": session_id,
                        "turn_index": turn_index,
                        "timestamp": event_time.isoformat(),
                    }
                    self._vector_only_add(text, user_id=user_id, run_id=session_id, metadata=metadata)
                else:
                    self._replace_l3_memory(
                        memory_id=memory_id,
                        new_text=text,
                        user_id=user_id,
                        session_id=session_id,
                        turn_index=turn_index,
                        timestamp=event_time,
                    )
                stored_count += 1

        return {"facts": facts, "actions": actions, "stored": stored_count}

    def save_l2_summary(
        self,
        user_id: str,
        session_id: str,
        l2_data: Dict[str, Any],
        start_time: datetime,
        end_time: datetime,
        turn_count: int,
    ) -> None:
        text = (
            f"主题：{l2_data.get('topic', '')}\n"
            f"背景：{l2_data.get('background', '')}\n"
            f"总结：{l2_data.get('summary', '')}"
        )
        metadata = {
            "memory_type": "l2_summary",
            "session_id": session_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "turn_count": turn_count,
            "topic": l2_data.get("topic", ""),
        }
        self._vector_only_add(text, user_id=user_id, run_id=session_id, metadata=metadata)

    def save_l3_fragment(
        self,
        user_id: str,
        session_id: str,
        turn_index: int,
        user_input: str,
        assistant_response: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self.remember_l3_turn(
            user_id=user_id,
            session_id=session_id,
            turn_index=turn_index,
            user_input=user_input,
            assistant_response=assistant_response,
            timestamp=timestamp,
        )

    def get_l2_summaries(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            with self._io_serialized():
                all_memories = self.client.get_all(user_id=user_id, limit=100)
        except Exception as exc:
            logger.error("Failed to get L2 summaries: %s", exc)
            return []

        summaries: List[Dict[str, Any]] = []
        for memory in all_memories.get("results", []):
            metadata = memory.get("metadata", {})
            if metadata.get("memory_type") != "l2_summary":
                continue
            summaries.append(
                {
                    "session_id": memory.get("run_id"),
                    "topic": metadata.get("topic"),
                    "summary": memory.get("memory"),
                    "start_time": metadata.get("start_time"),
                }
            )

        summaries.sort(key=lambda item: item.get("start_time", ""), reverse=True)
        return summaries[:limit]

    def search_relevant_context(self, user_id: str, query: str, limit: int = 3) -> Dict[str, Any]:
        search_limit = max(limit * 4, 12)
        vector_results: List[Dict[str, Any]] = []
        l2_summaries: List[Dict[str, Any]] = []
        session_graphs: List[Dict[str, Any]] = []
        graph_relations: List[Dict[str, Any]] = []

        try:
            with self._io_serialized():
                results = self.client.search(query, user_id=user_id, limit=search_limit)
            for item in results.get("results", []):
                metadata = item.get("metadata", {})
                memory_type = metadata.get("memory_type")
                payload = {
                    "memory": item.get("memory", ""),
                    "score": item.get("score"),
                    "session_id": metadata.get("session_id") or item.get("run_id"),
                }

                if memory_type == "l3_fragment":
                    payload["timestamp"] = metadata.get("timestamp")
                    payload["turn_index"] = metadata.get("turn_index")
                    vector_results.append(payload)
                elif memory_type == "l2_summary":
                    payload["topic"] = metadata.get("topic", "")
                    payload["start_time"] = metadata.get("start_time")
                    payload["summary"] = item.get("memory", "")
                    l2_summaries.append(payload)
                elif memory_type == "session_graph":
                    session_graphs.append(payload)

            for relation in results.get("relations", []) or []:
                if not isinstance(relation, dict):
                    continue
                graph_relations.append(
                    {
                        "source": relation.get("source", ""),
                        "relationship": relation.get("relationship", relation.get("relation", "")),
                        "target": relation.get("target", relation.get("destination", "")),
                    }
                )
        except Exception as exc:
            logger.error("Vector/graph search error: %s", exc)

        session_ids = list(
            {
                item["session_id"]
                for item in [*vector_results, *l2_summaries, *session_graphs]
                if item.get("session_id")
            }
        )
        if session_ids:
            try:
                with self._io_serialized():
                    all_memories = self.client.get_all(user_id=user_id, limit=100)
                for memory in all_memories.get("results", []):
                    metadata = memory.get("metadata", {})
                    if metadata.get("memory_type") != "l2_summary":
                        continue
                    if memory.get("run_id") not in session_ids:
                        continue
                    if any(item.get("session_id") == memory.get("run_id") for item in l2_summaries):
                        continue
                    l2_summaries.append(
                        {
                            "session_id": memory.get("run_id"),
                            "summary": memory.get("memory", ""),
                            "topic": metadata.get("topic", ""),
                            "score": None,
                            "start_time": metadata.get("start_time"),
                        }
                    )
            except Exception as exc:
                logger.error("Failed to fetch linked L2 summaries: %s", exc)

        l2_summaries.sort(
            key=lambda item: (
                item.get("score") is None,
                -(item.get("score") or 0),
                item.get("start_time") or "",
            )
        )
        vector_results.sort(key=lambda item: -(item.get("score") or 0))
        session_graphs.sort(key=lambda item: -(item.get("score") or 0))

        sections: List[str] = []
        if l2_summaries:
            sections.append("【相关历史会话摘要】")
            sections.extend(
                f"- {item.get('topic', '')}: {item.get('memory') or item.get('summary', '')}"
                for item in l2_summaries[:limit]
            )
        if vector_results:
            sections.append("【相关对话片段】")
            sections.extend(
                (
                    f"[{(item.get('timestamp') or '')[:10]}] {item.get('memory', '')}"
                    if item.get("timestamp")
                    else item.get("memory", "")
                )
                for item in vector_results[:limit]
            )
        if session_graphs:
            sections.append("【相关图记忆】")
            sections.extend(item.get("memory", "") for item in session_graphs[:limit])
        if graph_relations:
            sections.append("【相关图关系】")
            sections.extend(
                f"- {item.get('source', '')} --{item.get('relationship', '')}--> {item.get('target', '')}"
                for item in graph_relations[:limit]
            )

        return {
            "context_text": "\n".join(sections).strip(),
            "l3_fragments": vector_results[:limit],
            "l2_summaries": l2_summaries[:limit],
            "session_graphs": session_graphs[:limit],
            "graph_relations": graph_relations[:limit],
        }

    def search_memories(self, user_id: str, query: str, limit: int = 5) -> Dict[str, Any]:
        return self.search_relevant_context(user_id=user_id, query=query, limit=limit)

    def save_treatment_report(self, user_id: str, report: Dict[str, Any]) -> None:
        metadata = {
            "memory_type": "treatment_report",
            "timestamp": datetime.now().isoformat(),
        }
        self._vector_only_add(json.dumps(report, ensure_ascii=False), user_id=user_id, metadata=metadata)

    def get_latest_treatment_report(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._io_serialized():
                all_memories = self.client.get_all(user_id=user_id, limit=50)
            latest_memory: Optional[Dict[str, Any]] = None
            latest_timestamp: Optional[str] = None
            for memory in all_memories.get("results", []):
                metadata = memory.get("metadata", {})
                if metadata.get("memory_type") != "treatment_report":
                    continue
                timestamp = metadata.get("timestamp")
                if latest_timestamp is None or (timestamp and timestamp > latest_timestamp):
                    latest_timestamp = timestamp
                    latest_memory = memory
            if latest_memory:
                return json.loads(latest_memory.get("memory", "{}"))
        except Exception as exc:
            logger.error("Failed to get latest treatment report: %s", exc)
        return None
