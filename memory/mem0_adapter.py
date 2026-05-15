"""
Mem0 adapter for the multi_agent package.

This implementation uses real Mem0 vector retrieval together with the built-in
Kuzu graph backend. A local streaming patch is applied to Mem0's OpenAI LLM so
the graph pipeline can work with gateways that require `stream=true`.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterator, List, Optional

import torch
from dotenv import load_dotenv

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_DISABLED", "true")

_MEM0_IMPORT_ERROR: Optional[BaseException] = None

try:
    from mem0 import Memory
except BaseException as exc:  # pragma: no cover
    Memory = None  # type: ignore[assignment]
    _MEM0_IMPORT_ERROR = exc

from memory.mem0_stream_patch import patch_mem0_openai_streaming
from utils.model_client import ChatClientProtocol, build_chat_client, resolve_model_mode


logger = logging.getLogger(__name__)


def _is_safe_local_path(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        path = Path(text).expanduser()
    except Exception:
        return False
    if not path.is_absolute():
        return False
    return True


def _runtime_store_path(env_value: str, *, name: str, suffix: str = "") -> str:
    candidate = str(env_value or "").strip()
    if _is_safe_local_path(candidate):
        return str(Path(candidate).expanduser())
    root = Path(tempfile.gettempdir()) / "MemAgent"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{name}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    return str(target)


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
            detail = ""
            if _MEM0_IMPORT_ERROR is not None:
                detail = f" Original import error: {type(_MEM0_IMPORT_ERROR).__name__}: {_MEM0_IMPORT_ERROR}"
            raise ImportError(
                "mem0ai is not installed or could not be imported. "
                "Please install/package MemAgent memory dependencies before using Mem0Adapter."
                + detail
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
        self._graph_db_path = _runtime_store_path(os.getenv("KUZU_DB_PATH", ""), name="mem0_kuzu_db")
        self._chroma_db_path = _runtime_store_path(os.getenv("CHROMA_DB_PATH", ""), name="chroma_db")
        self._history_db_path = _runtime_store_path(os.getenv("MEM0_HISTORY_DB", ""), name="mem0_history", suffix=".db")
        self._graph_threshold = float(os.getenv("MEM0_GRAPH_THRESHOLD", "0.7"))
        self._retrieval_rrf_k = int(os.getenv("MEM0_RRF_K", "60"))
        self._retrieval_candidate_top_k = int(os.getenv("MEM0_RETRIEVAL_CANDIDATE_TOP_K", "20"))
        self._bm25_pool_limit = int(os.getenv("MEM0_BM25_POOL_LIMIT", "500"))
        self._bm25_enabled = os.getenv("MEM0_BM25_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        self._rrf_enabled = os.getenv("MEM0_RRF_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        self._reranker_setting = os.getenv("MEM0_RERANKER_ENABLED", "auto").strip().lower()
        self._reranker_model_ref = (
            os.getenv("MEM0_RERANKER_MODEL_PATH", "").strip()
            or os.getenv("MEM0_RERANKER_MODEL", "").strip()
        )
        self._reranker_tokenizer: Any = None
        self._reranker_model: Any = None
        self._reranker_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._reranker_load_error: Optional[str] = None
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
                    "path": self._chroma_db_path,
                },
            },
            "history_db_path": self._history_db_path,
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

    def _payload_from_memory_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        memory_type = metadata.get("memory_type", "")
        payload: Dict[str, Any] = {
            "id": item.get("id") or item.get("memory_id"),
            "memory": item.get("memory", ""),
            "score": item.get("score"),
            "memory_type": memory_type,
            "session_id": metadata.get("session_id") or item.get("run_id"),
        }
        if memory_type == "l3_fragment":
            payload["timestamp"] = metadata.get("timestamp")
            payload["turn_index"] = metadata.get("turn_index")
        elif memory_type == "graph_relation":
            payload["source"] = metadata.get("source", "")
            payload["relationship"] = metadata.get("relation", "")
            payload["target"] = metadata.get("destination", "")
        return payload

    @staticmethod
    def _retrieval_key(item: Dict[str, Any]) -> str:
        explicit_id = str(item.get("id") or "").strip()
        if explicit_id:
            return explicit_id
        return "|".join(
            [
                str(item.get("memory_type", "")),
                str(item.get("session_id", "")),
                str(item.get("memory", "")),
                str(item.get("source", "")),
                str(item.get("relationship", "")),
                str(item.get("target", "")),
            ]
        )

    @staticmethod
    def _tokenize_for_bm25(text: str) -> List[str]:
        text = str(text or "").lower()
        return re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", text)

    def _bm25_rank(self, query: str, candidates: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize_for_bm25(query)
        if not query_tokens or not candidates:
            return []

        docs = [self._tokenize_for_bm25(str(item.get("memory", ""))) for item in candidates]
        total_docs = len(docs)
        avg_len = sum(len(doc) for doc in docs) / max(total_docs, 1)
        if avg_len <= 0:
            return []

        doc_freq: Dict[str, int] = {}
        for doc in docs:
            for token in set(doc):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        k1 = 1.5
        b = 0.75
        scored: List[Dict[str, Any]] = []
        for item, doc in zip(candidates, docs):
            if not doc:
                continue
            freqs: Dict[str, int] = {}
            for token in doc:
                freqs[token] = freqs.get(token, 0) + 1
            score = 0.0
            doc_len = len(doc)
            for token in query_tokens:
                tf = freqs.get(token, 0)
                if not tf:
                    continue
                df = doc_freq.get(token, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1 - b + b * doc_len / avg_len)
                score += idf * (tf * (k1 + 1) / denom)
            if score > 0:
                payload = dict(item)
                payload["bm25_score"] = score
                scored.append(payload)
        scored.sort(key=lambda item: -float(item.get("bm25_score", 0) or 0))
        return scored[:limit]

    def _bm25_search(self, user_id: str, query: str, *, memory_type: str, limit: int) -> List[Dict[str, Any]]:
        if not self._bm25_enabled:
            return []
        try:
            with self._io_serialized(), self._graph_temporarily_disabled():
                all_memories = self.client.get_all(user_id=user_id, limit=self._bm25_pool_limit)
        except Exception as exc:
            logger.info("BM25 candidate load failed for user %s: %s", user_id, exc)
            return []

        candidates: List[Dict[str, Any]] = []
        for item in all_memories.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            payload = self._payload_from_memory_item(item)
            if payload.get("memory_type") == memory_type and str(payload.get("memory", "")).strip():
                candidates.append(payload)
        return self._bm25_rank(query, candidates, limit=limit)

    def _rrf_fuse(self, ranked_lists: List[List[Dict[str, Any]]], *, top_k: int) -> List[Dict[str, Any]]:
        if not self._rrf_enabled:
            merged: Dict[str, Dict[str, Any]] = {}
            for ranked in ranked_lists:
                for item in ranked:
                    key = self._retrieval_key(item)
                    current = merged.get(key)
                    if current is None or float(item.get("score") or item.get("bm25_score") or 0) > float(
                        current.get("score") or current.get("bm25_score") or 0
                    ):
                        merged[key] = dict(item)
            return sorted(merged.values(), key=lambda item: -float(item.get("score") or item.get("bm25_score") or 0))[:top_k]

        fused: Dict[str, Dict[str, Any]] = {}
        for source_index, ranked in enumerate(ranked_lists):
            source_name = "vector" if source_index == 0 else "bm25"
            for rank, item in enumerate(ranked, start=1):
                key = self._retrieval_key(item)
                record = fused.setdefault(key, dict(item))
                record["rrf_score"] = float(record.get("rrf_score", 0) or 0) + 1.0 / (self._retrieval_rrf_k + rank)
                sources = set(record.get("retrieval_sources", []))
                sources.add(source_name)
                record["retrieval_sources"] = sorted(sources)
                if item.get("score") is not None:
                    record["vector_score"] = item.get("score")
                if item.get("bm25_score") is not None:
                    record["bm25_score"] = item.get("bm25_score")
        return sorted(fused.values(), key=lambda item: -float(item.get("rrf_score", 0) or 0))[:top_k]

    def _resolve_reranker_model_ref(self) -> str:
        if self._reranker_model_ref:
            configured = Path(self._reranker_model_ref).expanduser()
            if configured.is_absolute():
                return str(configured)
            project_candidate = Path(__file__).resolve().parents[1] / configured
            if project_candidate.exists():
                return str(project_candidate.resolve())
            return self._reranker_model_ref
        project_default = Path(__file__).resolve().parents[1] / "models" / "bge-reranker-base"
        if project_default.exists():
            return str(project_default.resolve())
        return ""

    def _ensure_reranker(self) -> bool:
        if self._reranker_model is not None and self._reranker_tokenizer is not None:
            return True
        if self._reranker_load_error:
            return False
        model_ref = self._resolve_reranker_model_ref()
        if self._reranker_setting in {"0", "false", "no", "off"}:
            self._reranker_load_error = "disabled"
            return False
        if not model_ref and self._reranker_setting == "auto":
            self._reranker_load_error = "no local reranker configured"
            return False
        if not model_ref:
            model_ref = "BAAI/bge-reranker-base"
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_path = Path(model_ref).expanduser()
            local_files_only = model_path.exists()
            self._reranker_tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=local_files_only)
            self._reranker_model = AutoModelForSequenceClassification.from_pretrained(
                model_ref,
                local_files_only=local_files_only,
            ).to(self._reranker_device)
            self._reranker_model.eval()
            logger.info("Cross-Encoder reranker loaded: %s", model_ref)
            return True
        except Exception as exc:
            self._reranker_load_error = str(exc)
            logger.warning("Cross-Encoder reranker unavailable; using RRF order only: %s", exc)
            return False

    def _cross_encoder_rerank(self, query: str, candidates: List[Dict[str, Any]], *, top_k: int) -> List[Dict[str, Any]]:
        if not candidates or not self._ensure_reranker():
            return candidates[:top_k]
        try:
            pairs = [(query, str(item.get("memory", ""))) for item in candidates]
            scores: List[float] = []
            batch_size = int(os.getenv("MEM0_RERANKER_BATCH_SIZE", "8"))
            with torch.no_grad():
                for start in range(0, len(pairs), batch_size):
                    batch = pairs[start : start + batch_size]
                    inputs = self._reranker_tokenizer(
                        [pair[0] for pair in batch],
                        [pair[1] for pair in batch],
                        padding=True,
                        truncation=True,
                        max_length=int(os.getenv("MEM0_RERANKER_MAX_LENGTH", "512")),
                        return_tensors="pt",
                    )
                    inputs = {key: value.to(self._reranker_device) for key, value in inputs.items()}
                    logits = self._reranker_model(**inputs).logits
                    if logits.ndim == 2 and logits.shape[-1] > 1:
                        batch_scores = logits[:, -1]
                    else:
                        batch_scores = logits.reshape(-1)
                    scores.extend(float(score) for score in batch_scores.detach().cpu().tolist())
            reranked = []
            for item, score in zip(candidates, scores):
                payload = dict(item)
                payload["rerank_score"] = score
                reranked.append(payload)
            reranked.sort(key=lambda item: -float(item.get("rerank_score", 0) or 0))
            return reranked[:top_k]
        except Exception as exc:
            logger.warning("Cross-Encoder rerank failed; using RRF order only: %s", exc)
            return candidates[:top_k]

    def _ensure_l3_evidence(
        self,
        ranked_evidence: List[Dict[str, Any]],
        bm25_l3: List[Dict[str, Any]],
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if any(item.get("memory_type") == "l3_fragment" for item in ranked_evidence):
            return ranked_evidence[:limit]
        if not bm25_l3:
            return ranked_evidence[:limit]

        merged: List[Dict[str, Any]] = []
        seen = set()
        for item in [bm25_l3[0], *ranked_evidence]:
            key = self._retrieval_key(item)
            if key in seen:
                continue
            seen.add(key)
            payload = dict(item)
            sources = set(payload.get("retrieval_sources", []))
            sources.add("bm25")
            payload["retrieval_sources"] = sorted(sources)
            merged.append(payload)
            if len(merged) >= limit:
                break
        return merged

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
            logger.info(
                "Kuzu graph ingest is unavailable for session %s; vector graph memories remain available: %s",
                session_id,
                exc,
            )

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
        for relation in graph_data.get("relations", []) or []:
            if not isinstance(relation, dict):
                continue
            relation_text = (
                f"图关系: {relation.get('source', '')} --{relation.get('relation', '')}--> "
                f"{relation.get('destination', '')}"
            ).strip()
            if relation_text:
                self._vector_only_add(
                    relation_text,
                    user_id=user_id,
                    run_id=session_id,
                    metadata={
                        "memory_type": "graph_relation",
                        "source": relation.get("source", ""),
                        "relation": relation.get("relation", ""),
                        "destination": relation.get("destination", ""),
                    },
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

    def _print_retrieval_debug(
        self,
        *,
        user_id: str,
        query: str,
        l3_fragments: List[Dict[str, Any]],
        session_graphs: List[Dict[str, Any]],
        graph_relations: List[Dict[str, Any]],
        limit: int,
    ) -> None:
        enabled = os.getenv("MEMAGENT_DEBUG_RETRIEVAL", "1").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return

        print("\n" + "=" * 72, flush=True)
        print(f"[MemAgent Retrieval] user={user_id} limit={limit}", flush=True)
        print(f"[Query] {query}", flush=True)

        def _score_text(item: Dict[str, Any]) -> str:
            parts = []
            for key in ["rerank_score", "rrf_score", "vector_score", "bm25_score", "score"]:
                value = item.get(key)
                if value is not None:
                    try:
                        parts.append(f"{key}={float(value):.4f}")
                    except (TypeError, ValueError):
                        parts.append(f"{key}={value}")
            sources = item.get("retrieval_sources")
            if sources:
                parts.append(f"sources={','.join(str(source) for source in sources)}")
            return " ".join(parts) if parts else "score=N/A"

        print(f"\n[L3 片段记忆] count={len(l3_fragments[:limit])}", flush=True)
        for index, item in enumerate(l3_fragments[:limit], start=1):
            timestamp = (item.get("timestamp") or "")[:19]
            session_id = item.get("session_id") or ""
            print(
                f"{index}. {_score_text(item)} session={session_id} time={timestamp}\n"
                f"   {item.get('memory', '')}",
                flush=True,
            )

        print(f"\n[graph_relations 图关系] count={len(graph_relations[:limit])}", flush=True)
        for index, item in enumerate(graph_relations[:limit], start=1):
            if item.get("source") or item.get("target"):
                print(
                    f"{index}. {item.get('source', '')} --{item.get('relationship', '')}--> {item.get('target', '')}",
                    flush=True,
                )
            else:
                print(f"{index}. {item.get('memory', '')}", flush=True)
        print("=" * 72 + "\n", flush=True)

    def search_relevant_context(self, user_id: str, query: str, limit: int = 6, *, debug: bool = True) -> Dict[str, Any]:
        search_limit = max(limit * 4, 12)
        vector_results: List[Dict[str, Any]] = []
        session_graphs: List[Dict[str, Any]] = []
        graph_relations: List[Dict[str, Any]] = []

        try:
            with self._io_serialized():
                results = self.client.search(query, user_id=user_id, limit=search_limit)
            for item in results.get("results", []):
                metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
                memory_type = metadata.get("memory_type")
                payload = self._payload_from_memory_item(item)

                if memory_type == "l3_fragment":
                    vector_results.append(payload)
                elif memory_type == "session_graph":
                    session_graphs.append(payload)
                elif memory_type == "graph_relation":
                    graph_relations.append(payload)

            for relation in results.get("relations", []) or []:
                if not isinstance(relation, dict):
                    continue
                graph_relations.append(
                    {
                        "memory_type": "graph_relation",
                        "memory": (
                            f"图关系: {relation.get('source', '')} --"
                            f"{relation.get('relationship', relation.get('relation', ''))}--> "
                            f"{relation.get('target', relation.get('destination', ''))}"
                        ),
                        "source": relation.get("source", ""),
                        "relationship": relation.get("relationship", relation.get("relation", "")),
                        "target": relation.get("target", relation.get("destination", "")),
                    }
                )
        except Exception as exc:
            logger.info("Full vector/graph search failed; falling back to vector-only search: %s", exc)
            try:
                with self._io_serialized(), self._graph_temporarily_disabled():
                    fallback_results = self.client.search(query, user_id=user_id, limit=search_limit)
                for item in fallback_results.get("results", []):
                    metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
                    memory_type = metadata.get("memory_type")
                    payload = self._payload_from_memory_item(item)
                    if memory_type == "l3_fragment":
                        vector_results.append(payload)
                    elif memory_type == "session_graph":
                        session_graphs.append(payload)
                    elif memory_type == "graph_relation":
                        graph_relations.append(payload)
            except Exception as fallback_exc:
                logger.error("Fallback vector-only search failed: %s", fallback_exc)

        vector_results.sort(key=lambda item: -(item.get("score") or 0))
        session_graphs.sort(key=lambda item: -(item.get("score") or 0))
        graph_relations.sort(key=lambda item: -(item.get("score") or 0))

        bm25_l3 = self._bm25_search(user_id, query, memory_type="l3_fragment", limit=search_limit)
        bm25_graph_relations = self._bm25_search(user_id, query, memory_type="graph_relation", limit=search_limit)
        vector_evidence = vector_results + graph_relations
        bm25_evidence = bm25_l3 + bm25_graph_relations
        fused_evidence = self._rrf_fuse(
            [vector_evidence, bm25_evidence],
            top_k=self._retrieval_candidate_top_k,
        )
        reranked_evidence = self._cross_encoder_rerank(query, fused_evidence, top_k=limit)
        reranked_evidence = self._ensure_l3_evidence(reranked_evidence, bm25_l3, limit=limit)
        reranked_l3 = [item for item in reranked_evidence if item.get("memory_type") == "l3_fragment"]
        reranked_graph_relations = [item for item in reranked_evidence if item.get("memory_type") == "graph_relation"]

        if debug:
            self._print_retrieval_debug(
                user_id=user_id,
                query=query,
                l3_fragments=reranked_l3,
                session_graphs=session_graphs,
                graph_relations=reranked_graph_relations,
                limit=limit,
            )

        sections: List[str] = []
        if reranked_evidence:
            sections.append("【相关索引依据】")
            for index, item in enumerate(reranked_evidence[:limit], start=1):
                if item.get("memory_type") == "graph_relation" or item.get("source") or item.get("target"):
                    text = (
                        f"{item.get('source', '')} --{item.get('relationship', '')}--> {item.get('target', '')}"
                        if item.get("source") or item.get("target")
                        else str(item.get("memory", "") or "")
                    )
                else:
                    text = (
                        f"[{(item.get('timestamp') or '')[:10]}] {item.get('memory', '')}"
                        if item.get("timestamp")
                        else str(item.get("memory", "") or "")
                    )
                if text.strip():
                    sections.append(f"{index}. {text.strip()}")

        return {
            "context_text": "\n".join(sections).strip(),
            "l3_fragments": reranked_l3,
            "session_graphs": [],
            "graph_relations": reranked_graph_relations,
            "ranked_evidence": reranked_evidence,
            "retrieval_pipeline": {
                "vector_top_k": search_limit,
                "bm25_enabled": self._bm25_enabled,
                "rrf_enabled": self._rrf_enabled,
                "rrf_k": self._retrieval_rrf_k,
                "candidate_top_k": self._retrieval_candidate_top_k,
                "final_top_k": limit,
                "cross_encoder_enabled": self._ensure_reranker(),
                "cross_encoder_error": self._reranker_load_error or "",
                "l3_vector_candidates": len(vector_results),
                "l3_bm25_candidates": len(bm25_l3),
                "graph_vector_candidates": len(graph_relations),
                "graph_bm25_candidates": len(bm25_graph_relations),
                "global_vector_candidates": len(vector_evidence),
                "global_bm25_candidates": len(bm25_evidence),
                "global_fused_candidates": len(fused_evidence),
                "global_reranked_evidence": len(reranked_evidence),
            },
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
