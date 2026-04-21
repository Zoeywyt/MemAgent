import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv


def auto_load_dotenv(dotenv_path: Optional[str] = None) -> Optional[str]:
    if dotenv_path:
        load_dotenv(dotenv_path)
        return dotenv_path
    load_dotenv()
    return os.getenv("DOTENV_PATH")


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def iter_sse_data_lines(resp: object) -> Iterable[str]:
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data: "):
            continue
        yield line[6:]


def stream_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout: int,
    user_agent: str,
    print_stream: bool = False,
) -> Tuple[str, Optional[Dict[str, object]]]:
    endpoint = f"{normalize_base_url(base_url)}/chat/completions"
    payload = {
        "model": model,
        "stream": True,
        "temperature": 0.7,
        "messages": messages,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": user_agent,
        },
        method="POST",
    )

    pieces: List[str] = []
    usage: Optional[Dict[str, object]] = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for data_line in iter_sse_data_lines(resp):
                if data_line == "[DONE]":
                    break
                chunk = json.loads(data_line)
                if not isinstance(chunk, dict):
                    continue
                usage_obj = chunk.get("usage")
                if isinstance(usage_obj, dict):
                    usage = usage_obj
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if isinstance(content, str) and content:
                    pieces.append(content)
                    if print_stream:
                        sys.stdout.write(content)
                        sys.stdout.flush()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    if print_stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return "".join(pieces).strip(), usage


@dataclass
class OpenAIChatClient:
    base_url: str = os.getenv("OPENAI_BASE_URL", "")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("OPENAI_MODEL", "gpt-5.4")
    user_agent: str = os.getenv("OPENAI_USER_AGENT", "MultiAgent/1.0")
    timeout: int = 120

    def __post_init__(self):
        auto_load_dotenv()
        self.base_url = self.base_url or os.getenv("OPENAI_BASE_URL", "")
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = self.model or os.getenv("OPENAI_MODEL", "gpt-5.4")
        self.user_agent = self.user_agent or os.getenv("OPENAI_USER_AGENT", "MultiAgent/1.0")

        if not self.base_url or not self.api_key:
            raise ValueError("OPENAI_BASE_URL and OPENAI_API_KEY must be set in environment or .env")

    def chat(self, messages: List[Dict[str, str]], print_stream: bool = False) -> Tuple[str, Optional[Dict[str, object]]]:
        return stream_chat_completion(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            timeout=self.timeout,
            user_agent=self.user_agent,
            print_stream=print_stream,
        )
