import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from dotenv import load_dotenv


def _repair_bom_env_keys() -> None:
    for key in list(os.environ.keys()):
        if key.startswith("\ufeff"):
            normalized = key.lstrip("\ufeff")
            if normalized and normalized not in os.environ:
                os.environ[normalized] = os.environ[key]


def auto_load_dotenv(dotenv_path: Optional[str] = None) -> Optional[str]:
    if dotenv_path:
        load_dotenv(dotenv_path, encoding="utf-8-sig")
        _repair_bom_env_keys()
        return dotenv_path
    load_dotenv(encoding="utf-8-sig")
    _repair_bom_env_keys()
    return os.getenv("DOTENV_PATH")


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def iter_sse_data_lines(resp: object) -> Iterable[str]:
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data: "):
            continue
        yield line[6:]


def open_url(req: urllib.request.Request, timeout: int):
    """Open requests without inheriting broken shell proxy settings by default."""
    if os.getenv("OPENAI_USE_ENV_PROXY", "").strip() == "1":
        return urllib.request.urlopen(req, timeout=timeout)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req, timeout=timeout)


@dataclass(frozen=True)
class APIEndpointConfig:
    provider: str
    label: str
    base_url: str
    api_key: str
    model: str


REMOTE_API_FALLBACK_PRESETS: Dict[str, Dict[str, str]] = {
    "gpt": {
        "label": "GPT",
        "base_url_env": "OPENAI_BASE_URL",
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.4",
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "qwen": {
        "label": "Qwen",
        "base_url_env": "QWEN_BASE_URL",
        "api_key_env": "QWEN_API_KEY",
        "model_env": "QWEN_MODEL",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "kimi": {
        "label": "Kimi",
        "base_url_env": "KIMI_BASE_URL",
        "api_key_env": "KIMI_API_KEY",
        "model_env": "KIMI_MODEL",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
    },
}


def _looks_like_api_endpoint_error(exc: BaseException) -> bool:
    if isinstance(exc, RuntimeError):
        text = str(exc).lower()
        return (
            "api stream connection interrupted" in text
            or "http " in text
            or "connection" in text
            or "timed out" in text
            or "timeout" in text
        )
    return isinstance(exc, (ConnectionResetError, TimeoutError, urllib.error.URLError, OSError))


def _dedupe_endpoint_configs(configs: Iterable[APIEndpointConfig]) -> List[APIEndpointConfig]:
    deduped: List[APIEndpointConfig] = []
    seen = set()
    for config in configs:
        if not config.base_url or not config.api_key or not config.model:
            continue
        key = (normalize_base_url(config.base_url), config.api_key, config.model)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(config)
    return deduped


def _env_endpoint_configs() -> List[APIEndpointConfig]:
    configs: List[APIEndpointConfig] = []
    for provider, preset in REMOTE_API_FALLBACK_PRESETS.items():
        base_url = os.getenv(preset["base_url_env"], "").strip() or preset["base_url"]
        api_key = os.getenv(preset["api_key_env"], "").strip()
        model = os.getenv(preset["model_env"], "").strip() or preset["model"]
        if api_key:
            configs.append(
                APIEndpointConfig(
                    provider=provider,
                    label=preset["label"],
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                )
            )
    return configs


def stream_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout: int,
    user_agent: str,
    max_tokens: Optional[int] = None,
    print_stream: bool = False,
) -> Tuple[str, Optional[Dict[str, object]]]:
    pieces: List[str] = []
    for content in iter_stream_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        timeout=timeout,
        user_agent=user_agent,
        max_tokens=max_tokens,
    ):
        pieces.append(content)
        if print_stream:
            sys.stdout.write(content)
            sys.stdout.flush()

    if print_stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return "".join(pieces).strip(), None


def iter_stream_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout: int,
    user_agent: str,
    max_tokens: Optional[int] = None,
) -> Iterator[str]:
    endpoint = f"{normalize_base_url(base_url)}/chat/completions"
    payload = {
        "model": model,
        "stream": True,
        "temperature": 0.7,
        "messages": messages,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
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

    max_attempts = max(1, int(os.getenv("OPENAI_STREAM_RETRIES", "2")))
    last_error: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        yielded_any = False
        try:
            with open_url(req, timeout=timeout) as resp:
                for data_line in iter_sse_data_lines(resp):
                    if data_line == "[DONE]":
                        return
                    chunk = json.loads(data_line)
                    if not isinstance(chunk, dict):
                        continue
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
                        yielded_any = True
                        yield content
                return
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except (ConnectionResetError, TimeoutError, urllib.error.URLError, OSError) as exc:
            last_error = exc
            if yielded_any or attempt >= max_attempts:
                raise RuntimeError(
                    f"API stream connection interrupted after {attempt}/{max_attempts} attempt(s): {exc}"
                ) from exc
            continue
    if last_error is not None:
        raise RuntimeError(f"API stream connection interrupted: {last_error}") from last_error


@dataclass
class OpenAIChatClient:
    base_url: str = os.getenv("OPENAI_BASE_URL", "")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("OPENAI_MODEL", "gpt-5.4")
    provider: str = "gpt"
    user_agent: str = os.getenv("OPENAI_USER_AGENT", "MultiAgent/1.0")
    timeout: int = 120
    max_tokens: int = 1024

    def __post_init__(self):
        auto_load_dotenv()
        self.last_fallback_notice = ""
        self.last_provider = self.provider or "gpt"
        self.base_url = self.base_url or os.getenv("OPENAI_BASE_URL", "")
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = self.model or os.getenv("OPENAI_MODEL", "gpt-5.4")
        self.user_agent = self.user_agent or os.getenv("OPENAI_USER_AGENT", "MultiAgent/1.0")
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", str(self.max_tokens)))

        if not self.base_url or not self.api_key:
            raise ValueError("OPENAI_BASE_URL and OPENAI_API_KEY must be set in environment or .env")

    def _primary_endpoint(self) -> APIEndpointConfig:
        preset = REMOTE_API_FALLBACK_PRESETS.get(self.provider or "gpt", REMOTE_API_FALLBACK_PRESETS["gpt"])
        return APIEndpointConfig(
            provider=self.provider or "gpt",
            label=preset["label"],
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )

    def _candidate_endpoints(self) -> List[APIEndpointConfig]:
        if os.getenv("MEMAGENT_API_FALLBACK_ENABLED", "1").strip().lower() in {"0", "false", "no"}:
            return [self._primary_endpoint()]
        primary = self._primary_endpoint()
        env_configs = _env_endpoint_configs()
        other_providers = [item for item in env_configs if item.provider != primary.provider]
        same_provider = [item for item in env_configs if item.provider == primary.provider]
        return _dedupe_endpoint_configs([primary] + other_providers + same_provider)

    def _set_endpoint_notice(self, endpoint: APIEndpointConfig, primary: APIEndpointConfig) -> None:
        self.last_provider = endpoint.provider
        if (
            normalize_base_url(endpoint.base_url) != normalize_base_url(primary.base_url)
            or endpoint.model != primary.model
            or endpoint.api_key != primary.api_key
        ):
            self.last_fallback_notice = f"模型连接异常，已自动切换到 {endpoint.label}。"
            print(f"[MemAgent API Fallback] {self.last_fallback_notice}", flush=True)
        else:
            self.last_fallback_notice = ""

    def chat(self, messages: List[Dict[str, str]], print_stream: bool = False) -> Tuple[str, Optional[Dict[str, object]]]:
        primary = self._primary_endpoint()
        last_error: Optional[BaseException] = None
        for endpoint in self._candidate_endpoints():
            try:
                result = stream_chat_completion(
                    base_url=endpoint.base_url,
                    api_key=endpoint.api_key,
                    model=endpoint.model,
                    messages=messages,
                    timeout=self.timeout,
                    user_agent=self.user_agent,
                    max_tokens=self.max_tokens,
                    print_stream=print_stream,
                )
                self._set_endpoint_notice(endpoint, primary)
                return result
            except Exception as exc:
                last_error = exc
                if not _looks_like_api_endpoint_error(exc):
                    raise
                print(f"[MemAgent API Fallback] {endpoint.label} failed: {exc}", flush=True)
                continue
        raise RuntimeError(f"All configured API endpoints failed: {last_error}") from last_error

    def stream_chat(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        primary = self._primary_endpoint()
        last_error: Optional[BaseException] = None
        for endpoint in self._candidate_endpoints():
            yielded_any = False
            try:
                for piece in iter_stream_chat_completion(
                    base_url=endpoint.base_url,
                    api_key=endpoint.api_key,
                    model=endpoint.model,
                    messages=messages,
                    timeout=self.timeout,
                    user_agent=self.user_agent,
                    max_tokens=self.max_tokens,
                ):
                    if not yielded_any:
                        self._set_endpoint_notice(endpoint, primary)
                    yielded_any = True
                    yield piece
                return
            except Exception as exc:
                last_error = exc
                if yielded_any or not _looks_like_api_endpoint_error(exc):
                    raise
                print(f"[MemAgent API Fallback] {endpoint.label} failed before streaming: {exc}", flush=True)
                continue
        raise RuntimeError(f"All configured API endpoints failed: {last_error}") from last_error
