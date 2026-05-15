from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol, Tuple


ChatMessages = List[Dict[str, str]]


class ChatClientProtocol(Protocol):
    def chat(
        self,
        messages: ChatMessages,
        print_stream: bool = False,
    ) -> Tuple[str, Optional[Dict[str, object]]]:
        ...

    def stream_chat(self, messages: ChatMessages) -> Iterator[str]:
        ...


class ModelCallError(RuntimeError):
    pass


def _log_path() -> Path:
    return Path(__file__).resolve().parents[1] / "test_outputs" / "model_calls.log"


def _safe_message_preview(messages: ChatMessages, limit: int = 240) -> str:
    if not messages:
        return ""
    text = "\n".join(str(item.get("content", "")) for item in messages[-2:])
    text = " ".join(text.split())
    return text[:limit]


def log_model_event(
    *,
    component: str,
    event: str,
    elapsed: Optional[float] = None,
    error: str = "",
    notice: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "time": datetime.now().isoformat(),
        "component": component,
        "event": event,
    }
    if elapsed is not None:
        payload["elapsed_seconds"] = round(float(elapsed), 4)
    if error:
        payload["error"] = error
    if notice:
        payload["notice"] = notice
    if extra:
        payload.update(extra)
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def call_model(
    client: ChatClientProtocol,
    *,
    component: str,
    messages: ChatMessages,
    print_stream: bool = False,
) -> Tuple[str, Optional[Dict[str, object]]]:
    started = time.perf_counter()
    log_model_event(
        component=component,
        event="start",
        extra={"message_preview": _safe_message_preview(messages)},
    )
    try:
        response, usage = client.chat(messages=messages, print_stream=print_stream)
        elapsed = time.perf_counter() - started
        notice = str(getattr(client, "last_fallback_notice", "") or "")
        log_model_event(component=component, event="success", elapsed=elapsed, notice=notice)
        print(f"[MemAgent Model] {component} success in {elapsed:.2f}s", flush=True)
        return response, usage
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        log_model_event(
            component=component,
            event="error",
            elapsed=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )
        print(f"[MemAgent Model] {component} failed in {elapsed:.2f}s: {type(exc).__name__}: {exc}", flush=True)
        raise ModelCallError(f"{component} 模型调用失败：{exc}") from exc


def stream_model(
    client: ChatClientProtocol,
    *,
    component: str,
    messages: ChatMessages,
) -> Iterator[str]:
    started = time.perf_counter()
    yielded_any = False
    log_model_event(
        component=component,
        event="stream_start",
        extra={"message_preview": _safe_message_preview(messages)},
    )
    try:
        for piece in client.stream_chat(messages):
            if piece:
                yielded_any = True
                yield piece
        elapsed = time.perf_counter() - started
        notice = str(getattr(client, "last_fallback_notice", "") or "")
        log_model_event(component=component, event="stream_success", elapsed=elapsed, notice=notice)
        print(f"[MemAgent Model] {component} stream success in {elapsed:.2f}s", flush=True)
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        log_model_event(
            component=component,
            event="stream_error",
            elapsed=elapsed,
            error=f"{type(exc).__name__}: {exc}",
            extra={"yielded_any": yielded_any},
        )
        print(f"[MemAgent Model] {component} stream failed in {elapsed:.2f}s: {type(exc).__name__}: {exc}", flush=True)
        raise ModelCallError(f"{component} 模型流式调用失败：{exc}") from exc
