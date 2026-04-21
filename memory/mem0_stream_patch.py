from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from mem0.memory.utils import extract_json


logger = logging.getLogger(__name__)


def _streaming_generate_response(
    self,
    messages: List[Dict[str, str]],
    response_format: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    **kwargs: Any,
):
    params = self._get_supported_params(messages=messages, **kwargs)
    params.update(
        {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
        }
    )

    if os.getenv("OPENROUTER_API_KEY"):
        openrouter_params: Dict[str, Any] = {}
        if self.config.models:
            openrouter_params["models"] = self.config.models
            openrouter_params["route"] = self.config.route
            params.pop("model", None)

        if self.config.site_url and self.config.app_name:
            openrouter_params["extra_headers"] = {
                "HTTP-Referer": self.config.site_url,
                "X-Title": self.config.app_name,
            }

        params.update(**openrouter_params)
    else:
        for param in ["store"]:
            if hasattr(self.config, param):
                params[param] = getattr(self.config, param)

    if response_format:
        params["response_format"] = response_format
    if tools:
        params["tools"] = tools
        params["tool_choice"] = tool_choice

    stream = self.client.chat.completions.create(**params)

    content_parts: List[str] = []
    tool_call_parts: Dict[int, Dict[str, Any]] = {}

    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            content_parts.append(content)

        for tool_call in getattr(delta, "tool_calls", None) or []:
            index = getattr(tool_call, "index", 0) or 0
            current = tool_call_parts.setdefault(
                index,
                {
                    "name": "",
                    "arguments_parts": [],
                },
            )
            function = getattr(tool_call, "function", None)
            if function is None:
                continue
            name = getattr(function, "name", None)
            if isinstance(name, str) and name:
                current["name"] = name
            arguments = getattr(function, "arguments", None)
            if isinstance(arguments, str) and arguments:
                current["arguments_parts"].append(arguments)

    parsed_response: Any
    if tools:
        parsed_response = {
            "content": "".join(content_parts) or None,
            "tool_calls": [],
        }
        for index in sorted(tool_call_parts):
            item = tool_call_parts[index]
            raw_arguments = "".join(item["arguments_parts"])
            try:
                arguments = json.loads(extract_json(raw_arguments))
            except Exception:
                logger.warning("Failed to parse streamed tool arguments from Mem0 LLM: %r", raw_arguments)
                arguments = {}
            parsed_response["tool_calls"].append(
                {
                    "name": item["name"],
                    "arguments": arguments,
                }
            )
    else:
        parsed_response = "".join(content_parts)

    callback = getattr(self.config, "response_callback", None)
    if callback:
        try:
            callback(self, {"streamed": True, "parsed_response": parsed_response}, params)
        except Exception as exc:
            logger.error("Error due to callback: %s", exc)

    return parsed_response


def patch_mem0_openai_streaming() -> None:
    from mem0.llms.openai import OpenAILLM

    if getattr(OpenAILLM, "_multi_agent_stream_patch_applied", False):
        return

    OpenAILLM.generate_response = _streaming_generate_response
    OpenAILLM._multi_agent_stream_patch_applied = True
    logger.info("Applied streaming patch to mem0 OpenAILLM.generate_response")
