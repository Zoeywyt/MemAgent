from __future__ import annotations

import json
import html
import hmac
import os
import re
import sqlite3
import socket
import sys
import time
import hashlib
import secrets
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

CURRENT_DIR = Path(__file__).resolve().parent
TESTS_DIR = CURRENT_DIR / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from agents.empathy_agent import EmpathyAgent
from memory.user_summary_store import UserSummaryStore
from output_store import (
    CONSULTING_REPORTS_DIR,
    USER_REGISTRY_DB_PATH,
    USER_REGISTRY_JSON_PATH,
    USER_REGISTRY_PATH,
    consulting_report_html_path,
    consulting_report_json_path,
    ensure_output_dirs,
    load_user_sessions,
    migrate_legacy_session_tree,
    record_session_outputs,
    rename_user_session,
    load_user_session_summaries,
    sessions_summaries_path,
    sessions_path,
    load_user_trends,
    trends_path,
    whole_summary_path,
    WHOLE_SUMMARIES_DIR,
    safe_user_id,
)
from test_case import build_result_bundle
from utils.model_client import LOCAL_MODEL_PRESETS
from utils.openai_client import auto_load_dotenv


ensure_output_dirs()
REPORT_DIR = CONSULTING_REPORTS_DIR
BACKGROUND_FINALIZE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memagent-finalize")
BACKGROUND_FINALIZE_LOCK = threading.Lock()
BACKGROUND_FINALIZE_JOBS: Dict[str, Dict[str, Any]] = {}
MEMORY_WARMUP_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memagent-memory-warmup")
MEMORY_WARMUP_LOCK = threading.Lock()
MEMORY_WARMUP_FUTURE = None
PHONE_CODE_LOCK = threading.Lock()
PHONE_VERIFICATION_CODES: Dict[str, Dict[str, Any]] = {}
PHONE_CODE_TTL_SECONDS = int(os.getenv("MEMAGENT_PHONE_CODE_TTL_SECONDS", "300"))

MODEL_BACKEND_CHOICES = [
    ("GPT API", "gpt"),
    ("DeepSeek", "deepseek"),
    ("Qwen", "qwen"),
    ("Kimi", "kimi"),
    ("Qwen2.5-3B + 3B LoRA", "qwen3b"),
    ("Qwen2.5-7B + 7B LoRA", "qwen7b"),
]

AGENT_BACKEND_CHOICES = MODEL_BACKEND_CHOICES

SUMMARY_BACKEND_CHOICES = [
    ("GPT API", "gpt"),
    ("Qwen2.5-3B + 3B LoRA", "qwen3b"),
]

DEFAULT_API_MODEL_NAME = "gpt-5.4"
DEFAULT_API_PROVIDER = "gpt"
TRIAL_SESSION_LIMIT = int(os.getenv("MEMAGENT_TRIAL_SESSION_LIMIT", "5"))

API_PROVIDER_CHOICES = [
    ("GPT", "gpt"),
    ("DeepSeek", "deepseek"),
    ("Qwen", "qwen"),
    ("Kimi", "kimi"),
]

API_PROVIDER_PRESETS = {
    "gpt": {
        "base_url_env": "OPENAI_BASE_URL",
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "base_url": "",
        "model": "gpt-5.4",
    },
    "deepseek": {
        "base_url_env": "DEEPSEEK_BASE_URL",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "qwen": {
        "base_url_env": "QWEN_BASE_URL",
        "api_key_env": "QWEN_API_KEY",
        "model_env": "QWEN_MODEL",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "kimi": {
        "base_url_env": "KIMI_BASE_URL",
        "api_key_env": "KIMI_API_KEY",
        "model_env": "KIMI_MODEL",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
    },
}

API_MODE_CHOICES = [
    ("使用自己的 API Key", "own"),
    (f"试用阶段：免费 {TRIAL_SESSION_LIMIT} 次会话", "trial"),
]

DEFAULT_EXISTING_USER_PASSWORDS = {
    "CP224_刘某": "123",
    "刘某": "123",
    "Zoey": "456",
    "CP1033_小杰及陪同母亲": "789",
    "小杰": "789",
    "wyt": "888",
}


USER_REGISTRY_COLUMNS = (
    "user_id",
    "display_name",
    "created_at",
    "last_login_at",
    "source",
    "password_hash",
    "password_salt",
    "password_set_at",
    "password_updated_at",
    "phone_number",
    "phone_verified_at",
    "config_agent_backend",
    "config_summary_backend",
    "config_supervisor_backend",
    "config_graph_backend",
    "config_router_backend",
    "config_api_mode",
    "config_api_provider",
    "config_api_key",
    "config_api_base_url",
    "config_model",
    "config_model_keys",
    "trial_sessions_used",
)

DEFAULT_USER_CONFIG = {
    "config_agent_backend": "gpt",
    "config_summary_backend": "gpt",
    "config_supervisor_backend": "gpt",
    "config_graph_backend": "gpt",
    "config_router_backend": "gpt",
    "config_api_mode": "own",
    "config_api_provider": DEFAULT_API_PROVIDER,
    "config_api_key": "",
    "config_api_base_url": API_PROVIDER_PRESETS[DEFAULT_API_PROVIDER]["base_url"],
    "config_model": DEFAULT_API_MODEL_NAME,
    "config_model_keys": "{}",
}

REMOTE_API_PRESETS = API_PROVIDER_PRESETS


def _connect_user_registry_db() -> sqlite3.Connection:
    ensure_output_dirs()
    conn = sqlite3.connect(str(USER_REGISTRY_DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    # Some Windows folders fail when SQLite creates sidecar WAL/rollback journal
    # files. The account registry is tiny, so an in-memory journal is sufficient
    # and keeps login independent from the memory/output JSON files.
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=OFF")
    return conn


def _ensure_user_registry_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT '',
            last_login_at TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT 'legacy_import',
            password_hash TEXT NOT NULL DEFAULT '',
            password_salt TEXT NOT NULL DEFAULT '',
            password_set_at TEXT NOT NULL DEFAULT '',
            password_updated_at TEXT NOT NULL DEFAULT '',
            phone_number TEXT NOT NULL DEFAULT '',
            phone_verified_at TEXT NOT NULL DEFAULT '',
            config_agent_backend TEXT NOT NULL DEFAULT 'gpt',
            config_summary_backend TEXT NOT NULL DEFAULT 'gpt',
            config_supervisor_backend TEXT NOT NULL DEFAULT 'gpt',
            config_graph_backend TEXT NOT NULL DEFAULT 'gpt',
            config_router_backend TEXT NOT NULL DEFAULT 'gpt',
            config_api_mode TEXT NOT NULL DEFAULT 'own',
            config_api_provider TEXT NOT NULL DEFAULT 'gpt',
            config_api_key TEXT NOT NULL DEFAULT '',
            config_api_base_url TEXT NOT NULL DEFAULT '',
            config_model TEXT NOT NULL DEFAULT 'gpt-5.4',
            config_model_keys TEXT NOT NULL DEFAULT '{}',
            trial_sessions_used INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    for column in ["phone_number", "phone_verified_at"]:
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE users ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
    for column, default_value in DEFAULT_USER_CONFIG.items():
        if column not in existing_columns:
            quoted_default = "'" + str(default_value).replace("'", "''") + "'"
            conn.execute(
                f"ALTER TABLE users ADD COLUMN {column} TEXT NOT NULL DEFAULT {quoted_default}",
            )
    if "config_api_base_url" not in existing_columns and "config_api_base_url" not in DEFAULT_USER_CONFIG:
        conn.execute("ALTER TABLE users ADD COLUMN config_api_base_url TEXT NOT NULL DEFAULT ''")
    if "trial_sessions_used" not in existing_columns:
        conn.execute("ALTER TABLE users ADD COLUMN trial_sessions_used INTEGER NOT NULL DEFAULT 0")


def _profile_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    return {column: row[column] for column in USER_REGISTRY_COLUMNS}


def _default_user_config() -> Dict[str, str]:
    return dict(DEFAULT_USER_CONFIG)


def _normalize_api_provider(provider: str) -> str:
    value = (provider or DEFAULT_API_PROVIDER).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "gpt": "gpt",
        "openai": "gpt",
        "deepseek": "deepseek",
        "ds": "deepseek",
        "qwen": "qwen",
        "qwenapi": "qwen",
        "kimi": "kimi",
        "moonshot": "kimi",
    }
    return aliases.get(value, DEFAULT_API_PROVIDER)


def _api_provider_settings(provider: str) -> Dict[str, str]:
    auto_load_dotenv()
    normalized = _normalize_api_provider(provider)
    preset = API_PROVIDER_PRESETS.get(normalized, API_PROVIDER_PRESETS[DEFAULT_API_PROVIDER])
    base_url = os.getenv(str(preset.get("base_url_env", "")) or "", "").strip() or str(preset.get("base_url", "") or "")
    model = os.getenv(str(preset.get("model_env", "")) or "", "").strip() or str(
        preset.get("model", DEFAULT_API_MODEL_NAME) or DEFAULT_API_MODEL_NAME
    )
    return {
        "provider": normalized,
        "base_url": base_url,
        "model": model,
        "api_key_env": str(preset.get("api_key_env", "") or ""),
    }


def _api_provider_env_api_key(provider: str) -> str:
    settings = _api_provider_settings(provider)
    api_key_env = settings.get("api_key_env", "")
    if not api_key_env:
        return ""
    return os.getenv(api_key_env, "").strip()


def _api_provider_has_server_key(provider: str) -> bool:
    settings = _api_provider_settings(provider)
    return bool(settings.get("base_url", "").strip() and _api_provider_env_api_key(provider))


def _apply_api_provider_config(config: Dict[str, str]) -> Dict[str, str]:
    normalized = dict(config)
    settings = _api_provider_settings(normalized.get("config_api_provider", DEFAULT_API_PROVIDER))
    normalized["config_api_provider"] = settings["provider"]
    normalized["config_api_base_url"] = settings["base_url"]
    normalized["config_model"] = settings["model"]
    return normalized


def _load_model_key_map(raw_value: Any) -> Dict[str, str]:
    if isinstance(raw_value, dict):
        source = raw_value
    elif isinstance(raw_value, str) and raw_value.strip():
        try:
            parsed = json.loads(raw_value)
        except Exception:
            parsed = {}
        source = parsed if isinstance(parsed, dict) else {}
    else:
        source = {}

    key_map: Dict[str, str] = {}
    for key, value in source.items():
        normalized_key = _normalize_model_backend(str(key))
        key_value = str(value or "").strip()
        if key_value:
            key_map[normalized_key] = key_value
    return key_map


def _dump_model_key_map(key_map: Dict[str, str]) -> str:
    normalized: Dict[str, str] = {}
    for key, value in key_map.items():
        key_name = _normalize_model_backend(str(key))
        key_value = str(value or "").strip()
        if key_value:
            normalized[key_name] = key_value
    return json.dumps(normalized, ensure_ascii=False)


def _model_choice_settings(choice: str) -> Dict[str, str]:
    normalized = _normalize_model_backend(choice)
    if normalized in REMOTE_API_PRESETS:
        settings = _api_provider_settings(normalized)
        return {
            "choice": normalized,
            "backend": normalized,
            "provider": settings["provider"],
            "base_url": settings["base_url"],
            "model": settings["model"],
            "api_key_env": settings["api_key_env"],
            "requires_key": "1",
        }
    if normalized in LOCAL_MODEL_PRESETS:
        preset = LOCAL_MODEL_PRESETS.get(normalized, {})
        return {
            "choice": normalized,
            "backend": normalized,
            "provider": DEFAULT_API_PROVIDER,
            "base_url": "",
            "model": str(preset.get("model_path", "") or ""),
            "api_key_env": "",
            "requires_key": "0",
        }
    settings = _api_provider_settings(DEFAULT_API_PROVIDER)
    return {
        "choice": DEFAULT_API_PROVIDER,
        "backend": DEFAULT_API_PROVIDER,
        "provider": DEFAULT_API_PROVIDER,
        "base_url": settings["base_url"],
        "model": settings["model"],
        "api_key_env": settings["api_key_env"],
        "requires_key": "1",
    }


def _normalize_model_backend(backend: str) -> str:
    value = (backend or "gpt").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "gpt": "gpt",
        "openai": "gpt",
        "api": "gpt",
        "deepseek": "deepseek",
        "ds": "deepseek",
        "qwen": "qwen",
        "qwenapi": "qwen",
        "kimi": "kimi",
        "moonshot": "kimi",
        "qwen3b": "qwen3b",
        "qwen25o3b": "qwen3b",
        "qwen253b": "qwen3b",
        "3b": "qwen3b",
        "qwen7b": "qwen7b",
        "qwen25o7b": "qwen7b",
        "qwen257b": "qwen7b",
        "7b": "qwen7b",
    }
    return aliases.get(value, "gpt")


def _bundle_model_backends(backend: str) -> Dict[str, str]:
    normalized = _normalize_model_backend(backend)
    return {
        "config_model_backend": normalized,
        "config_agent_backend": normalized,
        "config_summary_backend": normalized,
        "config_supervisor_backend": normalized,
        "config_graph_backend": normalized,
        "config_router_backend": normalized,
    }


def _user_config_from_profile(profile: Dict[str, Any]) -> Dict[str, str]:
    config = _default_user_config()
    for key in config:
        value = str(profile.get(key, "") or "").strip()
        if value:
            config[key] = value
    if config.get("config_api_mode") not in {"own", "trial"}:
        config["config_api_mode"] = "own"
    selected_model = _normalize_model_backend(config.get("config_agent_backend", "gpt"))
    key_map = _load_model_key_map(config.get("config_model_keys", "{}"))
    legacy_key = str(config.get("config_api_key", "") or "").strip()
    if legacy_key and selected_model not in key_map:
        key_map[selected_model] = legacy_key
    settings = _model_choice_settings(selected_model)
    config.update(_bundle_model_backends(settings["choice"]))
    config["config_api_provider"] = settings["provider"]
    config["config_api_base_url"] = settings["base_url"]
    config["config_model"] = settings["model"]
    config["config_model_keys"] = _dump_model_key_map(key_map)
    config["config_api_key"] = key_map.get(settings["choice"], "")
    return config


def _upsert_user_profile(conn: sqlite3.Connection, profile: Dict[str, Any]) -> None:
    normalized = _normalize_user_entry(dict(profile))
    normalized.setdefault("password_updated_at", "")
    values = [str(normalized.get(column, "") or "") for column in USER_REGISTRY_COLUMNS]
    columns_sql = ", ".join(USER_REGISTRY_COLUMNS)
    placeholders = ", ".join("?" for _ in USER_REGISTRY_COLUMNS)
    conn.execute(
        f"""
        INSERT INTO users ({columns_sql})
        VALUES ({placeholders})
        ON CONFLICT(user_id) DO UPDATE SET
            display_name=excluded.display_name,
            created_at=excluded.created_at,
            last_login_at=excluded.last_login_at,
            source=excluded.source,
            password_hash=excluded.password_hash,
            password_salt=excluded.password_salt,
            password_set_at=excluded.password_set_at,
            password_updated_at=excluded.password_updated_at,
            phone_number=excluded.phone_number,
            phone_verified_at=excluded.phone_verified_at,
            config_agent_backend=excluded.config_agent_backend,
            config_summary_backend=excluded.config_summary_backend,
            config_supervisor_backend=excluded.config_supervisor_backend,
            config_graph_backend=excluded.config_graph_backend,
            config_router_backend=excluded.config_router_backend,
            config_api_mode=excluded.config_api_mode,
            config_api_provider=excluded.config_api_provider,
            config_api_key=excluded.config_api_key,
            config_api_base_url=excluded.config_api_base_url,
            config_model=excluded.config_model,
            config_model_keys=excluded.config_model_keys,
            trial_sessions_used=excluded.trial_sessions_used
        """,
        values,
    )


def _load_legacy_user_registry_json() -> Dict[str, Any]:
    if not USER_REGISTRY_JSON_PATH.exists():
        return {"users": []}
    try:
        registry = json.loads(USER_REGISTRY_JSON_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"users": []}
    if not isinstance(registry, dict):
        return {"users": []}
    users = registry.get("users", [])
    return {"users": users if isinstance(users, list) else []}


def _migrate_legacy_user_registry_if_needed(conn: sqlite3.Connection) -> None:
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count or not USER_REGISTRY_JSON_PATH.exists():
        return
    legacy_registry = _load_legacy_user_registry_json()
    for item in legacy_registry.get("users", []):
        if isinstance(item, dict) and item.get("user_id"):
            _upsert_user_profile(conn, item)


def load_user_registry() -> Dict[str, Any]:
    with _connect_user_registry_db() as conn:
        _ensure_user_registry_db(conn)
        _migrate_legacy_user_registry_if_needed(conn)
        rows = conn.execute(
            "SELECT user_id, display_name, created_at, last_login_at, source, "
            "password_hash, password_salt, password_set_at, password_updated_at, "
            "phone_number, phone_verified_at, "
            "config_agent_backend, config_summary_backend, config_supervisor_backend, "
            "config_graph_backend, config_router_backend, config_api_mode, config_api_provider, "
            "config_api_key, config_api_base_url, config_model, config_model_keys, trial_sessions_used "
            "FROM users ORDER BY user_id"
        ).fetchall()
    return {"users": [_profile_from_row(row) for row in rows]}


def save_user_registry(registry: Dict[str, Any]) -> None:
    users = registry.get("users", []) if isinstance(registry, dict) else []
    with _connect_user_registry_db() as conn:
        _ensure_user_registry_db(conn)
        for item in users:
            if isinstance(item, dict) and item.get("user_id"):
                _upsert_user_profile(conn, item)


def _normalize_user_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    entry.setdefault("display_name", entry.get("user_id", ""))
    entry.setdefault("created_at", "")
    entry.setdefault("last_login_at", "")
    entry.setdefault("source", "legacy_import")
    entry.setdefault("password_hash", "")
    entry.setdefault("password_salt", "")
    entry.setdefault("password_set_at", "")
    entry.setdefault("password_updated_at", "")
    entry.setdefault("phone_number", "")
    entry.setdefault("phone_verified_at", "")
    for key, value in DEFAULT_USER_CONFIG.items():
        entry.setdefault(key, value)
    selected_model = _normalize_model_backend(str(entry.get("config_agent_backend", "gpt") or "gpt"))
    key_map = _load_model_key_map(entry.get("config_model_keys", "{}"))
    current_key = str(entry.get("config_api_key", "") or "").strip()
    if current_key and selected_model not in key_map:
        key_map[selected_model] = current_key
    entry["config_model_keys"] = _dump_model_key_map(key_map)
    entry["config_api_key"] = key_map.get(selected_model, current_key)
    entry.setdefault("trial_sessions_used", 0)
    return entry


def _password_digest(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return digest, salt


def _verify_password(profile: Dict[str, Any], password: str) -> bool:
    stored_hash = str(profile.get("password_hash") or "")
    stored_salt = str(profile.get("password_salt") or "")
    if not stored_hash or not stored_salt:
        return False
    digest, _ = _password_digest(password, stored_salt)
    return hmac.compare_digest(digest, stored_hash)


def _set_password(profile: Dict[str, Any], password: str) -> None:
    digest, salt = _password_digest(password)
    profile["password_hash"] = digest
    profile["password_salt"] = salt
    profile["password_set_at"] = datetime.now().isoformat()


def _default_password_for_profile(profile: Dict[str, Any]) -> str:
    user_id = str(profile.get("user_id") or "")
    display_name = str(profile.get("display_name") or "")
    for marker, password in DEFAULT_EXISTING_USER_PASSWORDS.items():
        if marker and (marker == user_id or marker == display_name or marker in user_id or marker in display_name):
            return password
    return ""


def _form_message(text: str = "", kind: str = "error") -> str:
    if not text:
        return ""
    kind_class = "form-ok" if kind == "ok" else "form-error"
    return f'<div class="{kind_class}">{_html_escape(text)}</div>'


def _normalize_phone_number(phone: str) -> str:
    digits = re.sub(r"\D+", "", str(phone or ""))
    if digits.startswith("86") and len(digits) == 13:
        digits = digits[2:]
    return digits


def _masked_phone(phone: str) -> str:
    digits = _normalize_phone_number(phone)
    if len(digits) >= 7:
        return f"{digits[:3]}****{digits[-4:]}"
    return digits


def _validate_phone_number(phone: str) -> Tuple[str, str]:
    digits = _normalize_phone_number(phone)
    if not digits:
        return "", "请输入手机号。"
    if not re.fullmatch(r"1\d{10}", digits):
        return "", "手机号格式不正确，请输入 11 位中国大陆手机号。"
    return digits, ""


def _find_profile_by_phone(phone: str) -> Optional[Dict[str, Any]]:
    digits = _normalize_phone_number(phone)
    if not digits:
        return None
    registry = load_user_registry()
    for item in registry.get("users", []):
        if not isinstance(item, dict):
            continue
        _normalize_user_entry(item)
        if str(item.get("phone_number", "")) == digits and str(item.get("phone_verified_at", "")):
            return item
    return None


def _issue_phone_code(phone: str, purpose: str, user_id: str = "") -> str:
    code = f"{secrets.randbelow(1_000_000):06d}"
    expires_at = time.time() + PHONE_CODE_TTL_SECONDS
    with PHONE_CODE_LOCK:
        PHONE_VERIFICATION_CODES[_normalize_phone_number(phone)] = {
            "code": code,
            "purpose": purpose,
            "user_id": user_id,
            "expires_at": expires_at,
        }
    print(
        f"[MemAgent Phone Code] phone={_masked_phone(phone)} purpose={purpose} user={user_id or '-'} code={code}",
        flush=True,
    )
    return code


def _verify_phone_code(phone: str, code: str, purpose: str, user_id: str = "") -> Tuple[bool, str]:
    digits = _normalize_phone_number(phone)
    code = str(code or "").strip()
    with PHONE_CODE_LOCK:
        payload = PHONE_VERIFICATION_CODES.get(digits)
        if not payload:
            return False, "验证码不存在或已过期，请重新获取。"
        if float(payload.get("expires_at", 0) or 0) < time.time():
            PHONE_VERIFICATION_CODES.pop(digits, None)
            return False, "验证码已过期，请重新获取。"
        if payload.get("purpose") != purpose:
            return False, "验证码用途不匹配，请重新获取。"
        if user_id and payload.get("user_id") and payload.get("user_id") != user_id:
            return False, "验证码账号不匹配，请重新获取。"
        if str(payload.get("code", "")) != code:
            return False, "验证码错误，请检查后重试。"
        PHONE_VERIFICATION_CODES.pop(digits, None)
    return True, ""


def _server_trial_api_available(provider: str = DEFAULT_API_PROVIDER) -> bool:
    auto_load_dotenv()
    settings = _api_provider_settings(provider)
    return bool(settings.get("base_url", "").strip() and _api_provider_env_api_key(provider))


def _trial_used_from_profile(profile: Dict[str, Any]) -> int:
    try:
        return max(0, int(profile.get("trial_sessions_used", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _trial_quota_text(profile_or_state: Dict[str, Any]) -> str:
    used = _trial_used_from_profile(profile_or_state)
    remaining = max(TRIAL_SESSION_LIMIT - used, 0)
    return f"试用额度：已使用 {used}/{TRIAL_SESSION_LIMIT} 次会话，剩余 {remaining} 次。"


def _trial_quota_message(profile_or_state: Dict[str, Any]) -> str:
    return _form_message(_trial_quota_text(profile_or_state), "ok")


def _find_registry_user(user_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    registry = load_user_registry()
    for item in registry.get("users", []):
        if isinstance(item, dict) and item.get("user_id") == user_id:
            _normalize_user_entry(item)
            return registry, item
    return registry, {}


def _safe_sanitize_user_id(raw_value: str) -> Tuple[str, str]:
    text = (raw_value or "").strip()
    if not text:
        return "", "请输入用户名或用户 ID。"
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return "", "用户 ID 不合法，请换一个名称。"
    return text, ""


def sanitize_user_id(raw_value: str) -> str:
    text = (raw_value or "").strip()
    if not text:
        raise gr.Error("请输入用户名或用户 ID。")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        raise gr.Error("用户 ID 不合法，请换一个名称。")
    return text


def collect_known_user_ids() -> List[str]:
    user_ids = set()
    registry = load_user_registry()
    for item in registry.get("users", []):
        if isinstance(item, dict) and item.get("user_id"):
            user_ids.add(str(item["user_id"]))

    summary_dir = Path("multi_agent_user_summaries")
    if summary_dir.exists():
        for path in summary_dir.glob("*_summary.json"):
            user_ids.add(path.name[: -len("_summary.json")])

    if WHOLE_SUMMARIES_DIR.exists():
        for path in WHOLE_SUMMARIES_DIR.glob("*_summary.json"):
            user_ids.add(path.name[: -len("_summary.json")])

    return sorted(user_ids)


def default_login_user_id(choices: List[str]) -> Optional[str]:
    if "CP224_刘某" in choices:
        return "CP224_刘某"
    for user_id in choices:
        if "刘某" in user_id:
            return user_id
    return choices[0] if choices else None


def resolve_existing_user_id(raw_value: str) -> str:
    text = (raw_value or "").strip()
    if not text:
        return ""
    profile = find_user_profile(text)
    if profile:
        return str(profile.get("user_id") or text)
    registry = load_user_registry()
    for item in registry.get("users", []):
        if not isinstance(item, dict):
            continue
        user_id = str(item.get("user_id") or "")
        display_name = str(item.get("display_name") or "")
        if text == display_name or text in user_id or text in display_name:
            return user_id
    return sanitize_user_id(text)


def ensure_seed_users() -> None:
    registry = load_user_registry()
    known = {item.get("user_id", "") for item in registry.get("users", []) if isinstance(item, dict)}
    changed = False
    for user_id in collect_known_user_ids():
        if user_id and user_id not in known:
            registry.setdefault("users", []).append(
                _normalize_user_entry(
                    {
                        "user_id": user_id,
                        "display_name": user_id,
                        "created_at": datetime.now().isoformat(),
                        "last_login_at": "",
                        "source": "seeded",
                    }
                )
            )
            changed = True
    for item in registry.get("users", []):
        if isinstance(item, dict):
            before = dict(item)
            _normalize_user_entry(item)
            default_password = _default_password_for_profile(item)
            if default_password and not _verify_password(item, default_password):
                _set_password(item, default_password)
            if item != before:
                changed = True
    if changed or not USER_REGISTRY_PATH.exists():
        save_user_registry(registry)


def find_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    registry = load_user_registry()
    for item in registry.get("users", []):
        if isinstance(item, dict) and item.get("user_id") == user_id:
            return item
    return None


def register_user(user_id: str, display_name: str, password: str) -> str:
    user_id = sanitize_user_id(user_id)
    if not password.strip():
        raise gr.Error("请为新账户设置密码。")
    if find_user_profile(user_id):
        raise gr.Error(f"用户 `{user_id}` 已存在，请直接登录。")

    registry = load_user_registry()
    profile = _normalize_user_entry(
        {
            "user_id": user_id,
            "display_name": display_name.strip() or user_id,
            "created_at": datetime.now().isoformat(),
            "last_login_at": "",
            "source": "manual_register",
        }
    )
    _set_password(profile, password)
    registry.setdefault("users", []).append(profile)
    save_user_registry(registry)

    summary_store = UserSummaryStore()
    if summary_store.get_user_summary(user_id) is None:
        summary_store.save_user_summary(
            user_id,
            {"主题": "未知", "背景": "用户信息不足", "会话总结": "暂无对话历史"},
        )
    return user_id


def build_agent(
    agent_backend: str,
    summary_backend: str,
    supervisor_backend: str,
    graph_backend: str,
    router_backend: str,
    *,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> EmpathyAgent:
    return EmpathyAgent(
        model_backend=agent_backend,
        summary_model_backend=summary_backend,
        supervisor_model_backend=supervisor_backend,
        graph_model_backend=graph_backend,
        router_model_backend=router_backend,
        api_key=api_key or None,
        base_url=base_url or None,
        model=model or None,
    )


def _resolved_api_runtime_config(config: Dict[str, Any]) -> Dict[str, str]:
    api_mode = str(config.get("api_mode", "own") or "own").strip().lower()
    selection = _normalize_model_backend(
        str(
            config.get("model_backend", config.get("api_provider", DEFAULT_API_PROVIDER))
            or DEFAULT_API_PROVIDER
        )
    )
    settings = _model_choice_settings(selection)
    if settings["choice"] in LOCAL_MODEL_PRESETS:
        return {
            "api_mode": api_mode,
            "api_key": "",
            "base_url": "",
            "model": "",
            "api_provider": settings["provider"],
        }
    if api_mode == "trial":
        auto_load_dotenv()
        return {
            "api_mode": "trial",
            "api_key": _api_provider_env_api_key(settings["provider"]),
            "base_url": settings["base_url"],
            "model": settings["model"],
            "api_provider": settings["provider"],
        }
    return {
        "api_mode": "own",
        "api_key": str(config.get("api_key", "") or "").strip() or _api_provider_env_api_key(settings["provider"]),
        "base_url": str(config.get("api_base_url", "") or settings["base_url"]).strip() or settings["base_url"],
        "model": str(config.get("model", "") or settings["model"]).strip() or settings["model"],
        "api_provider": settings["provider"],
    }


def update_model_selection_preview(
    model_backend: str,
    api_mode: str = "own",
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[dict, dict]:
    selection = _normalize_model_backend(model_backend)
    settings = _model_choice_settings(selection)
    api_mode = str(api_mode or "own").strip().lower()
    key_map = {}
    if isinstance(state, dict):
        config = state.get("config", {})
        if isinstance(config, dict):
            key_map = _load_model_key_map(config.get("model_keys", config.get("config_model_keys", {})))
    key_value = "" if api_mode == "trial" or selection in LOCAL_MODEL_PRESETS else key_map.get(selection, "")
    model_label = settings["model"] or ""
    return gr.update(value=key_value), gr.update(value=model_label)


def empty_state() -> Dict[str, Any]:
    return {
        "authenticated": False,
        "user_id": "",
        "display_name": "",
        "agent": None,
        "chatbot": [],
        "l3_records": [],
        "retrieval_records": [],
        "current_session_id": "",
        "active_chatbot_start_index": 0,
        "preloaded_session_context": {},
        "config": {},
        "agent_ready": False,
        "hub_section": "overview",
        "history_session_id": "",
        "background_finalize_jobs": [],
        "background_finalize_message": "",
        "trial_sessions_used": 0,
        "trial_charged_session_id": "",
    }


def _background_job_status_label(stage: str) -> str:
    labels = {
        "queued": "已进入后台队列",
        "graph": "正在抽取图谱关系",
        "summary": "正在生成本次会话总结",
        "l1": "正在更新长期画像",
        "report": "正在生成督导师报告",
        "persist": "正在写入 test_outputs",
        "done": "已完成后台归档",
        "error": "后台归档失败",
    }
    return labels.get(stage, stage or "处理中")


def _register_background_finalize_job(snapshot: Dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex
    with BACKGROUND_FINALIZE_LOCK:
        BACKGROUND_FINALIZE_JOBS[job_id] = {
            "job_id": job_id,
            "user_id": snapshot["user_id"],
            "run_session_id": snapshot["run_session_id"],
            "stage": "queued",
            "detail": "",
            "status": "queued",
            "error": "",
            "done": False,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "result": {},
        }
    return job_id


def _update_background_finalize_job(job_id: str, *, stage: str, detail: str = "", done: bool = False, error: str = "", result: Optional[Dict[str, Any]] = None) -> None:
    with BACKGROUND_FINALIZE_LOCK:
        job = BACKGROUND_FINALIZE_JOBS.get(job_id)
        if not job:
            return
        job["stage"] = stage
        job["detail"] = detail
        job["status"] = "done" if done else ("error" if error else "running")
        job["error"] = error
        job["done"] = done
        if result is not None:
            job["result"] = result
        job["updated_at"] = datetime.now().isoformat()


def _get_background_finalize_job(job_id: str) -> Optional[Dict[str, Any]]:
    with BACKGROUND_FINALIZE_LOCK:
        job = BACKGROUND_FINALIZE_JOBS.get(job_id)
        return dict(job) if job else None


def _format_background_finalize_status(job: Dict[str, Any], queue_size: int) -> str:
    status = _background_job_status_label(str(job.get("stage", "")))
    detail = str(job.get("detail", "") or "")
    suffix = f"\n{detail}" if detail else ""
    queue_text = f"\n后台队列中还有 {max(queue_size - 1, 0)} 个任务。" if queue_size > 1 else ""
    return f"后台归档：{status}{suffix}{queue_text}"


def _session_snapshot_from_state(state: Dict[str, Any], agent: "EmpathyAgent") -> Dict[str, Any]:
    session_messages = [dict(item) for item in getattr(agent, "session_messages", [])]
    chat_pairs = []
    for idx in range(0, len(session_messages), 2):
        user_msg = session_messages[idx]
        assistant_msg = session_messages[idx + 1] if idx + 1 < len(session_messages) else {"content": ""}
        if user_msg.get("role") == "user":
            chat_pairs.append({"user": user_msg.get("content", ""), "assistant": assistant_msg.get("content", "")})
    return {
        "user_id": state["user_id"],
        "display_name": state["display_name"],
        "run_session_id": state["current_session_id"],
        "chat_pairs": chat_pairs,
        "session_messages": session_messages,
        "session_start_time": getattr(agent, "session_start_time", None) or datetime.now(),
    }


def _format_snapshot_session_for_graph(session_messages: List[Dict[str, str]]) -> str:
    lines = []
    turn = 1
    for idx in range(0, len(session_messages), 2):
        user_msg = session_messages[idx]
        assistant_msg = session_messages[idx + 1] if idx + 1 < len(session_messages) else None
        lines.append(f"轮次{turn} user：{user_msg.get('content', '')}")
        if assistant_msg:
            lines.append(f"轮次{turn} assistant：{assistant_msg.get('content', '')}")
        turn += 1
    return "\n".join(lines)


def _run_background_finalize_job(job_id: str, snapshot: Dict[str, Any], agent: "EmpathyAgent") -> None:
    try:
        _update_background_finalize_job(job_id, stage="graph", detail="正在抽取图谱关系")
        conversation_text = _format_snapshot_session_for_graph(snapshot["session_messages"])
        graph_data = agent.graph_extractor.extract_graph(conversation_text, user_id=snapshot["user_id"])
        if agent.enable_mem0_runtime:
            agent.mem0.add_graph_data(
                snapshot["user_id"],
                snapshot["run_session_id"],
                graph_data,
                source_text=conversation_text,
            )

        _update_background_finalize_job(job_id, stage="summary", detail="正在生成本次会话总结")
        l2_summary = agent.summary_agent.generate_l2_summary(snapshot["session_messages"])
        if agent.enable_mem0_runtime:
            agent.mem0.save_l2_summary(
                snapshot["user_id"],
                snapshot["run_session_id"],
                l2_summary,
                snapshot["session_start_time"],
                datetime.now(),
                len(snapshot["session_messages"]) // 2,
            )

        _update_background_finalize_job(job_id, stage="l1", detail="正在更新长期画像")
        new_l1 = agent.summary_agent.update_l1_summary(snapshot["user_id"], l2_summary)
        summary_doc = load_user_session_summaries(snapshot["user_id"])
        all_l2 = summary_doc.get("sessions", []) if isinstance(summary_doc.get("sessions", []), list) else []
        all_l2.insert(
            0,
            {
                "session_id": snapshot["run_session_id"],
                "topic": l2_summary.get("topic", ""),
                "summary": l2_summary.get("summary", ""),
                "start_time": snapshot["session_start_time"].isoformat() if hasattr(snapshot["session_start_time"], "isoformat") else datetime.now().isoformat(),
            },
        )
        _update_background_finalize_job(job_id, stage="report", detail="正在生成督导师报告")
        report = agent.supervisor.generate_treatment_report(new_l1, all_l2)
        if agent.enable_mem0_runtime:
            agent.mem0.save_treatment_report(snapshot["user_id"], report)

        _update_background_finalize_job(job_id, stage="persist", detail="正在写入 test_outputs")
        output_paths = record_session_outputs(
            user_id=snapshot["user_id"],
            display_name=snapshot["display_name"],
            run_session_id=snapshot["run_session_id"],
            chat_pairs=snapshot["chat_pairs"],
            l2_summary=l2_summary,
            treatment_report=report,
            l1_summary=new_l1,
            archived_at=datetime.now().isoformat(),
        )
        _update_background_finalize_job(
            job_id,
            stage="done",
            detail="后台归档完成",
            done=True,
            result={
                "graph_data": graph_data,
                "l2_summary": l2_summary,
                "l1_summary": new_l1,
                "treatment_report": report,
                "output_paths": output_paths,
            },
        )
    except Exception as exc:
        with BACKGROUND_FINALIZE_LOCK:
            job = BACKGROUND_FINALIZE_JOBS.get(job_id)
            if job is not None:
                job["error"] = str(exc)
                job["stage"] = "error"
                job["status"] = "error"
                job["done"] = True
                job["updated_at"] = datetime.now().isoformat()
        _update_background_finalize_job(job_id, stage="error", detail=str(exc), error=str(exc), done=True)


def _schedule_background_finalize(state: Dict[str, Any], agent: "EmpathyAgent") -> str:
    snapshot = _session_snapshot_from_state(state, agent)
    job_id = _register_background_finalize_job(snapshot)
    BACKGROUND_FINALIZE_EXECUTOR.submit(_run_background_finalize_job, job_id, snapshot, agent)
    pending = list(state.get("background_finalize_jobs", []))
    pending.append(job_id)
    state["background_finalize_jobs"] = pending
    state["background_finalize_message"] = f"后台归档已排队：{snapshot['run_session_id']}"
    return job_id


def refresh_background_finalize_state(state: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str, str]:
    if not state.get("authenticated") or not state.get("user_id"):
        return state, "", "", ""

    pending = list(state.get("background_finalize_jobs", []))
    status_text = str(state.get("background_finalize_message", "") or "")
    status_update = gr.skip()
    memory_html = gr.skip()
    retrieval_html = gr.skip()

    if pending:
        job_id = pending[0]
        job = _get_background_finalize_job(job_id)
        if job:
            status_text = _format_background_finalize_status(job, len(pending))
            status_update = status_text
            if str(job.get("status", "")) == "error":
                status_text = f"后台归档失败：{job.get('error', '')}"
                status_update = status_text
                state["background_finalize_message"] = status_text
                try:
                    gr.Warning(status_text)
                except Exception:
                    pass
                pending.pop(0)
                state["background_finalize_jobs"] = pending
            elif job.get("done"):
                pending.pop(0)
                state["background_finalize_jobs"] = pending
                result = job.get("result", {}) if isinstance(job.get("result", {}), dict) else {}
                if result:
                    latest_summary = result.get("l1_summary", {})
                    latest_report = result.get("treatment_report", {})
                    state["preloaded_session_context"] = {
                        "l1_summary": latest_summary,
                        "treatment_report": latest_report,
                        "preloaded_context_text": "",
                    }
                    if state.get("agent") is not None:
                        state["agent"].preloaded_session_context = state["preloaded_session_context"]
                    if result.get("output_paths"):
                        state["last_output_paths"] = result["output_paths"]
                status_text = "后台归档已完成，最新会话总结和最新督导师报告已更新。"
                status_update = status_text
                state["background_finalize_message"] = status_text
                try:
                    gr.Info("后台归档已完成，最新会话总结和最新督导师报告已更新。")
                except Exception:
                    pass
                if pending:
                    status_text = f"{status_text}\n后台队列中还有 {len(pending)} 个任务。"
                    status_update = status_text
                memory_html = build_session_memory_panel(state["user_id"], state.get("display_name", ""))
                retrieval_html = render_retrieval_records(state.get("retrieval_records", []), state.get("display_name", ""))
        else:
            pending.pop(0)
            state["background_finalize_jobs"] = pending

    return state, status_update, memory_html, retrieval_html


def poll_background_finalize_state(state: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    if not isinstance(state, dict) or not state.get("authenticated") or not state.get("user_id"):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    if not state.get("background_finalize_jobs"):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    return refresh_background_finalize_state(state)


def _state_api_config(state: Dict[str, Any]) -> Dict[str, str]:
    config = state.get("config", {}) if isinstance(state.get("config", {}), dict) else {}
    return {
        "api_key": str(config.get("api_key", "") or ""),
        "model": str(config.get("model", "") or DEFAULT_API_MODEL_NAME),
        "api_provider": str(config.get("api_provider", DEFAULT_API_PROVIDER) or DEFAULT_API_PROVIDER),
        "api_base_url": str(config.get("api_base_url", "") or ""),
    }


def _needs_remote_api(*backends: str) -> bool:
    for backend in backends:
        normalized = _normalize_model_backend(backend)
        if normalized in REMOTE_API_PRESETS or normalized == "gpt":
            return True
    return False


def _require_api_key_if_needed(state: Dict[str, Any], *backends: str) -> Optional[str]:
    if not _needs_remote_api(*backends):
        return None
    config = state.get("config", {}) if isinstance(state.get("config", {}), dict) else {}
    provider = _normalize_api_provider(str(config.get("api_provider", DEFAULT_API_PROVIDER) or DEFAULT_API_PROVIDER))
    if str(config.get("api_mode", "own") or "own").strip().lower() == "trial":
        if _server_trial_api_available(provider):
            return None
        settings = _api_provider_settings(provider)
        api_key_env = settings.get("api_key_env", "OPENAI_API_KEY")
        base_url = settings.get("base_url", "")
        if provider == "gpt":
            return "服务器试用 Key 未配置，请先在环境变量或 .env 中设置 OPENAI_BASE_URL 和 OPENAI_API_KEY。"
        return f"服务器试用 Key 未配置，请先在环境变量或 .env 中设置 {base_url} 对应的 {api_key_env}。"
    api_key = str(config.get("api_key", "") or "").strip()
    if api_key:
        return None
    return "选择 API 模型时，请先在“我的配置”里填写自己的 API Key。"


def _login_success_response(profile: Dict[str, Any], user_id: str, message_prefix: str = "登录成功") -> Tuple[Any, ...]:
    _normalize_user_entry(profile)
    user_config = _user_config_from_profile(profile)
    state = empty_state()
    state["authenticated"] = True
    state["user_id"] = user_id
    state["display_name"] = str((profile.get("display_name") if profile else None) or user_id)
    phone_number = str(profile.get("phone_number", "") or "")
    phone_verified_at = str(profile.get("phone_verified_at", "") or "")
    state["phone_number"] = phone_number
    state["phone_verified_at"] = phone_verified_at
    effective_user_api_key = "" if user_config["config_api_mode"] == "trial" else user_config["config_api_key"]
    state["config"] = {
        "model_backend": user_config["config_model_backend"],
        "agent_backend": user_config["config_agent_backend"],
        "summary_backend": user_config["config_summary_backend"],
        "supervisor_backend": user_config["config_supervisor_backend"],
        "graph_backend": user_config["config_graph_backend"],
        "router_backend": user_config["config_router_backend"],
        "api_mode": user_config["config_api_mode"],
        "api_provider": user_config["config_api_provider"],
        "api_base_url": user_config["config_api_base_url"],
        "api_key": effective_user_api_key,
        "model": user_config["config_model"],
        "config_model_keys": user_config["config_model_keys"],
        "trial_sessions_used": _trial_used_from_profile(profile),
        "trial_charged_session_id": "",
    }
    archived_messages = archived_chatbot_messages(user_id)
    state["chatbot"] = archived_messages
    state["active_chatbot_start_index"] = len(archived_messages)
    state["retrieval_records"] = load_recent_retrieval_records(user_id)
    memory_html = build_session_memory_panel(user_id, state["display_name"])
    choices = session_choices(user_id)
    selected_session = latest_choice_value(choices)
    hub_html, _, meta = render_consultation_hub(user_id, "overview", selected_session)
    state["history_session_id"] = meta.get("history_session_id", selected_session)
    return (
        state,
        f"{message_prefix}：{state['display_name']} ({user_id})",
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
        state["display_name"],
        memory_html,
        dropdown_update(choices, selected_session, False),
        hub_html,
        gr.update(visible=False),
        "",
        gr.update(value=""),
        gr.update(value=f"我的咨询 · {state['display_name']}"),
        gr.update(value=user_config["config_supervisor_backend"]),
        gr.update(value=user_config["config_graph_backend"]),
        gr.update(value=user_config["config_router_backend"]),
        gr.update(value=user_config["config_agent_backend"]),
        gr.update(value=user_config["config_summary_backend"]),
        gr.update(value=user_config["config_router_backend"]),
        gr.update(value=user_config["config_api_provider"]),
        gr.update(value=effective_user_api_key),
        gr.update(value=user_config["config_model"]),
        gr.update(value=user_config["config_api_mode"]),
        _trial_quota_message(profile),
        gr.update(value=phone_number),
        _form_message(f"当前已绑定：{_masked_phone(phone_number)}。", "ok") if phone_number and phone_verified_at else "",
        render_retrieval_records(state.get("retrieval_records", []), state["display_name"]),
        gr.update(value=state["chatbot"], visible=True),
    )


def login_user(
    selected_user_id: str,
    typed_user_id: str,
    password: str,
) -> Tuple[Any, ...]:
    selected_user_id = (selected_user_id or "").strip()
    typed_user_id = (typed_user_id or "").strip()
    password = (password or "").strip()
    user_id = resolve_existing_user_id(typed_user_id) if typed_user_id else selected_user_id

    def fail(message: str) -> Tuple[Any, ...]:
        return (
            empty_state(),
            "登录失败。",
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            "",
            dropdown_update([], "", False),
            "",
            gr.update(visible=False),
            _form_message(message),
            gr.update(value=""),
            gr.update(value="我的咨询"),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=DEFAULT_API_PROVIDER),
            gr.update(value=""),
            gr.update(value=DEFAULT_API_MODEL_NAME),
            gr.update(value="own"),
            _trial_quota_message({"trial_sessions_used": 0}),
            gr.update(value=""),
            "",
            render_retrieval_records([], ""),
            gr.update(value=[], visible=True),
        )

    if not user_id:
        return fail("请输入用户名后再登录。")
    if not password:
        return fail("请输入密码。")

    profile = find_user_profile(user_id)
    if not profile:
        return fail("未找到该账号，请先注册账户。")

    registry = load_user_registry()
    updated = False
    for item in registry.get("users", []):
        if isinstance(item, dict) and item.get("user_id") == user_id:
            _normalize_user_entry(item)
            profile = item
            updated = True
            break
    if not profile:
        return fail("账号信息读取失败，请重新打开页面。")

    stored_hash = str(profile.get("password_hash") or "")
    stored_salt = str(profile.get("password_salt") or "")
    if stored_hash and stored_salt:
        if not _verify_password(profile, password):
            return fail("密码错误，请检查后重试。")
    else:
        _set_password(profile, password)
        updated = True

    now = datetime.now().isoformat()
    profile["last_login_at"] = now
    profile.setdefault("display_name", user_id)
    profile.setdefault("source", "legacy_import")
    updated = True
    if updated:
        save_user_registry(registry)
    return _login_success_response(profile, user_id, "登录成功")


def _active_chatbot_start(state: Dict[str, Any]) -> int:
    try:
        start = int(state.get("active_chatbot_start_index", 0) or 0)
    except (TypeError, ValueError):
        start = 0
    chatbot = state.get("chatbot", [])
    if not isinstance(chatbot, list):
        return 0
    return max(0, min(start, len(chatbot)))


def _previous_chatbot_messages(state: Dict[str, Any]) -> List[Dict[str, str]]:
    start = _active_chatbot_start(state)
    return _clean_chatbot_messages(state.get("chatbot", [])[:start])


def _active_chatbot_messages(state: Dict[str, Any]) -> List[Dict[str, str]]:
    start = _active_chatbot_start(state)
    return _clean_chatbot_messages(state.get("chatbot", [])[start:])


def register_and_refresh(user_id: str, display_name: str, password: str, confirm_password: str) -> Tuple[str, dict, dict, dict, dict, str, str, str, str]:
    user_id, error = _safe_sanitize_user_id(user_id)
    if error:
        return _form_message(error), gr.update(), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), "", display_name or "", "", ""
    if not password.strip():
        return _form_message("请为新账户设置密码。"), gr.update(), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), user_id, display_name or "", "", ""
    if password != (confirm_password or ""):
        return _form_message("两次输入的密码不一致。"), gr.update(), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), user_id, display_name or "", "", ""
    if find_user_profile(user_id):
        return _form_message(f"用户 `{user_id}` 已存在，请直接登录。"), gr.update(), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), user_id, display_name or "", "", ""
    try:
        user_id = register_user(user_id, display_name, password)
    except Exception as exc:
        return _form_message(f"注册失败：{exc}"), gr.update(), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), user_id, display_name or "", "", ""
    return (
        _form_message(f"注册成功：{user_id}，请返回登录。", "ok"),
        gr.update(choices=collect_known_user_ids(), value=None),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        "",
        "",
        "",
    )


def open_register_page() -> Tuple[dict, dict, dict, str]:
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), "请先注册账户，再返回登录。"


def open_login_page() -> Tuple[dict, dict, dict, str]:
    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), "已返回登录界面。"


def clear_login_password() -> Tuple[dict, str]:
    return gr.update(value=""), ""


def fill_login_user_from_dropdown(user_id: str) -> Tuple[dict, dict, str]:
    return gr.update(value=user_id or ""), gr.update(value=""), ""


def _choice_value(choice: Any) -> str:
    if isinstance(choice, (list, tuple)) and len(choice) >= 2:
        return str(choice[1])
    return str(choice or "")


def dropdown_value(value: str, choices: List[Any]) -> Optional[str]:
    value = normalize_session_id_choice(value)
    values = {_choice_value(choice) for choice in choices}
    return value if value and value in values else None


def latest_choice_value(choices: List[Any]) -> str:
    return _choice_value(choices[-1]) if choices else ""


def dropdown_update(choices: List[Any], value: str = "", visible: bool = False) -> dict:
    if not choices:
        return gr.update(choices=[""], value="", visible=False)
    return gr.update(choices=choices, value=dropdown_value(value, choices), visible=visible)


def change_password(
    state: Dict[str, Any],
    old_password: str,
    new_password: str,
    confirm_password: str,
) -> Tuple[str, str, str, str]:
    if not state.get("authenticated") or not state.get("user_id"):
        return _form_message("请先登录后再修改密码。"), "", "", ""
    old_password = (old_password or "").strip()
    new_password = (new_password or "").strip()
    confirm_password = (confirm_password or "").strip()
    if not old_password:
        return _form_message("请输入原密码。"), "", "", ""
    if not new_password:
        return _form_message("请输入新密码。"), "", "", ""
    if len(new_password) < 3:
        return _form_message("新密码至少需要 3 个字符。"), "", "", ""
    if new_password != confirm_password:
        return _form_message("两次输入的新密码不一致。"), "", "", ""

    registry = load_user_registry()
    for item in registry.get("users", []):
        if isinstance(item, dict) and item.get("user_id") == state["user_id"]:
            _normalize_user_entry(item)
            if not _verify_password(item, old_password):
                return _form_message("原密码错误，请重新输入。"), "", "", ""
            _set_password(item, new_password)
            item["password_updated_at"] = datetime.now().isoformat()
            save_user_registry(registry)
            return _form_message("密码修改成功。", "ok"), "", "", ""
    return _form_message("账号信息不存在，请重新登录。"), "", "", ""


def request_phone_login_code(phone: str) -> str:
    phone, error = _validate_phone_number(phone)
    if error:
        return _form_message(error)
    profile = _find_profile_by_phone(phone)
    if not profile:
        return _form_message("该手机号尚未绑定已验证账号，请先使用密码登录后绑定手机号。")
    code = _issue_phone_code(phone, "phone_login", str(profile.get("user_id", "")))
    return _form_message(f"验证码已生成。本地展示验证码：{code}，5 分钟内有效。", "ok")


def verify_phone_login(
    phone: str,
    code: str,
    new_password: str,
    confirm_password: str,
) -> Tuple[Any, ...]:
    def fail(message: str) -> Tuple[Any, ...]:
        return (
            empty_state(),
            "验证码登录失败。",
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            "",
            dropdown_update([], "", False),
            "",
            gr.update(visible=False),
            _form_message(message),
            gr.update(value=""),
            gr.update(value="我的咨询"),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=DEFAULT_API_PROVIDER),
            gr.update(value=""),
            gr.update(value=DEFAULT_API_MODEL_NAME),
            gr.update(value="own"),
            _trial_quota_message({"trial_sessions_used": 0}),
            gr.update(value=""),
            "",
            render_retrieval_records([], ""),
            gr.update(value=[], visible=True),
        )

    phone, error = _validate_phone_number(phone)
    if error:
        return fail(error)
    profile = _find_profile_by_phone(phone)
    if not profile:
        return fail("该手机号尚未绑定已验证账号。")
    ok, message = _verify_phone_code(phone, code, "phone_login", str(profile.get("user_id", "")))
    if not ok:
        return fail(message)

    new_password = (new_password or "").strip()
    confirm_password = (confirm_password or "").strip()
    registry = load_user_registry()
    updated_profile: Optional[Dict[str, Any]] = None
    for item in registry.get("users", []):
        if isinstance(item, dict) and item.get("user_id") == profile.get("user_id"):
            _normalize_user_entry(item)
            item["last_login_at"] = datetime.now().isoformat()
            if new_password or confirm_password:
                if len(new_password) < 3:
                    return fail("新密码至少需要 3 个字符。")
                if new_password != confirm_password:
                    return fail("两次输入的新密码不一致。")
                _set_password(item, new_password)
                item["password_updated_at"] = datetime.now().isoformat()
            updated_profile = item
            break
    if not updated_profile:
        return fail("账号信息不存在，请重新登录。")
    save_user_registry(registry)
    return _login_success_response(updated_profile, str(updated_profile.get("user_id", "")), "手机号验证登录成功")


def request_bind_phone_code(state: Dict[str, Any], phone: str) -> str:
    if not state.get("authenticated") or not state.get("user_id"):
        return _form_message("请先登录后再绑定手机号。")
    phone, error = _validate_phone_number(phone)
    if error:
        return _form_message(error)
    registry = load_user_registry()
    for item in registry.get("users", []):
        if not isinstance(item, dict):
            continue
        _normalize_user_entry(item)
        if item.get("phone_number") == phone and item.get("user_id") != state["user_id"]:
            return _form_message("该手机号已绑定其他账号。")
    code = _issue_phone_code(phone, "bind_phone", state["user_id"])
    return _form_message(f"验证码已生成。本地展示验证码：{code}，5 分钟内有效。", "ok")


def bind_phone_number(state: Dict[str, Any], phone: str, code: str) -> Tuple[Dict[str, Any], dict, str]:
    if not state.get("authenticated") or not state.get("user_id"):
        return state, gr.update(), _form_message("请先登录后再绑定手机号。")
    phone, error = _validate_phone_number(phone)
    if error:
        return state, gr.update(), _form_message(error)
    ok, message = _verify_phone_code(phone, code, "bind_phone", state["user_id"])
    if not ok:
        return state, gr.update(), _form_message(message)

    registry = load_user_registry()
    updated = False
    for item in registry.get("users", []):
        if not isinstance(item, dict):
            continue
        _normalize_user_entry(item)
        if item.get("phone_number") == phone and item.get("user_id") != state["user_id"]:
            return state, gr.update(), _form_message("该手机号已绑定其他账号。")
    for item in registry.get("users", []):
        if isinstance(item, dict) and item.get("user_id") == state["user_id"]:
            _normalize_user_entry(item)
            item["phone_number"] = phone
            item["phone_verified_at"] = datetime.now().isoformat()
            state["phone_number"] = item["phone_number"]
            state["phone_verified_at"] = item["phone_verified_at"]
            updated = True
            break
    if not updated:
        return state, gr.update(), _form_message("账号信息不存在，请重新登录。")
    save_user_registry(registry)
    return state, gr.update(value=phone), _form_message(f"手机号已绑定：{_masked_phone(phone)}。", "ok")


def save_model_config(
    state: Dict[str, Any],
    model_backend: str,
    api_mode: str,
    api_provider: str,
    api_key: str,
    model: str,
) -> Tuple[Dict[str, Any], str, dict, dict, str]:
    if not state.get("authenticated") or not state.get("user_id"):
        return state, _form_message("请先登录后再保存配置。"), gr.update(), gr.update(), _trial_quota_message(state)
    model_backend = _normalize_model_backend(model_backend)
    model_settings = _model_choice_settings(model_backend)
    api_mode = (api_mode or "own").strip().lower()
    if api_mode not in {"own", "trial"}:
        api_mode = "own"
    api_provider = model_settings["provider"]
    provider_settings = model_settings
    api_key = (api_key or "").strip()
    model = provider_settings["model"]
    base_url = provider_settings["base_url"]
    key_map = _load_model_key_map(state.get("config", {}).get("config_model_keys", "{}") if isinstance(state.get("config", {}), dict) else "{}")
    if _needs_remote_api(model_backend):
        if api_mode == "own" and not api_key and not _api_provider_env_api_key(api_provider):
            return state, _form_message("当前选择了 API 模型，请先填写自己的 API Key，或在 .env 中配置对应供应商的默认 Key。"), gr.update(), gr.update(), _trial_quota_message(state)
        if api_mode == "trial":
            if not _server_trial_api_available(api_provider):
                key_env = provider_settings.get("api_key_env", "OPENAI_API_KEY")
                return state, _form_message(f"服务器试用 Key 未配置，请先在 .env 中填写 {key_env}。"), gr.update(), gr.update(), _trial_quota_message(state)
            api_key = ""
        if api_key:
            key_map[model_backend] = api_key
    else:
        api_key = ""
    key_value = "" if api_mode == "trial" or model_backend in LOCAL_MODEL_PRESETS else key_map.get(model_backend, "")
    user_bundle = _bundle_model_backends(model_backend)
    user_config = {
        "model_backend": user_bundle["config_model_backend"],
        "agent_backend": user_bundle["config_agent_backend"],
        "summary_backend": user_bundle["config_summary_backend"],
        "supervisor_backend": user_bundle["config_supervisor_backend"],
        "graph_backend": user_bundle["config_graph_backend"],
        "router_backend": user_bundle["config_router_backend"],
        "api_mode": api_mode,
        "api_provider": api_provider,
        "api_base_url": base_url,
        "api_key": key_value,
        "model": model,
        "config_model_keys": _dump_model_key_map(key_map),
    }
    state["config"] = dict(user_config)
    registry = load_user_registry()
    updated = False
    for item in registry.get("users", []):
        if isinstance(item, dict) and item.get("user_id") == state["user_id"]:
            _normalize_user_entry(item)
            item["config_agent_backend"] = user_config["agent_backend"]
            item["config_summary_backend"] = user_config["summary_backend"]
            item["config_supervisor_backend"] = user_config["supervisor_backend"]
            item["config_graph_backend"] = user_config["graph_backend"]
            item["config_router_backend"] = user_config["router_backend"]
            item["config_api_mode"] = user_config["api_mode"]
            item["config_api_provider"] = user_config["api_provider"]
            item["config_api_key"] = user_config["api_key"]
            item["config_api_base_url"] = user_config["api_base_url"]
            item["config_model"] = user_config["model"]
            item["config_model_keys"] = user_config["config_model_keys"]
            updated = True
            state["trial_sessions_used"] = _trial_used_from_profile(item)
            break
    if not updated:
        return state, _form_message("账号信息不存在，配置未保存，请重新登录后再试。"), gr.update(), gr.update(), _trial_quota_message(state)
    save_user_registry(registry)
    if not state.get("chatbot"):
        state["agent"] = None
        state["agent_ready"] = False
        state["current_session_id"] = ""
        state["preloaded_session_context"] = {}
    key_update = gr.update(value=key_value)
    model_update = gr.update(value=model)
    mode_note = "配置已保存。" if api_mode == "trial" else "配置已保存。未开始的新会话会使用当前配置。"
    return state, _form_message(mode_note, "ok"), key_update, model_update, _trial_quota_message(state)


def logout_user() -> Tuple[Any, ...]:
    return (
        empty_state(),
        "已退出登录。",
        "",
        "",
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        "",
        "",
        dropdown_update([], "", False),
        "",
        "",
        "",
        "",
        "",
        gr.update(value="我的咨询"),
        render_retrieval_records([], ""),
    )


def _clean_chatbot_messages(value: Any) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if not isinstance(value, list):
        return messages
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "") or "")
        content = str(item.get("content", "") or "")
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": content})
    return messages


def _chatbot_to_session_messages(chatbot: List[Dict[str, str]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for item in chatbot:
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content", "") or "")
        if role == "assistant" and not content.strip():
            continue
        messages.append({"role": role, "content": content})
    return messages


def _session_messages_to_chatbot(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    chatbot: List[Dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content", "") or "")
        if role == "assistant" and not content.strip():
            continue
        chatbot.append({"role": role, "content": content})
    return chatbot


def _clean_retrieval_records(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for item in value[-5:]:
        if isinstance(item, dict):
            cleaned.append(item)
    return cleaned


def retrieval_records_path(user_id: str) -> Path:
    return CURRENT_DIR / "test_outputs" / "retrieval_evidence" / f"{safe_user_id(user_id)}.json"


def load_recent_retrieval_records(user_id: str) -> List[Dict[str, Any]]:
    payload = read_json_payload(retrieval_records_path(user_id), {"records": []})
    records = payload.get("records", []) if isinstance(payload, dict) else []
    return _clean_retrieval_records(records)


def save_recent_retrieval_records(user_id: str, records: List[Dict[str, Any]]) -> None:
    if not user_id:
        return
    path = retrieval_records_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"user_id": user_id, "updated_at": datetime.now().isoformat(), "records": _clean_retrieval_records(records)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def archived_chatbot_messages(user_id: str) -> List[Dict[str, str]]:
    doc = load_user_sessions(user_id)
    sessions = doc.get("sessions", []) if isinstance(doc.get("sessions", []), list) else []
    valid_sessions = [item for item in sessions if isinstance(item, dict)]
    if not valid_sessions:
        return []
    latest_session = max(
        valid_sessions,
        key=lambda item: (
            item.get("id", 0) if isinstance(item.get("id"), int) else 0,
            str(item.get("archived_at", "")),
            str(item.get("session_id", "")),
        ),
    )
    messages: List[Dict[str, str]] = []
    for turn in latest_session.get("turns", []) or []:
        if not isinstance(turn, dict):
            continue
        user_text = str(turn.get("user", "") or "").strip()
        assistant_text = str(turn.get("assistant", "") or "").strip()
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
    return messages


def _browser_session_payload(
    state: Dict[str, Any],
    *,
    page: str = "chat",
    hub_section: str = "overview",
    history_session_id: str = "",
) -> Dict[str, Any]:
    if not state.get("authenticated") or not state.get("user_id"):
        return {}
    agent = state.get("agent")
    session_start_time = getattr(agent, "session_start_time", None) if agent is not None else None
    if hasattr(session_start_time, "isoformat"):
        session_start_time = session_start_time.isoformat()
    else:
        session_start_time = ""
    active_start = _active_chatbot_start(state)
    chatbot = state.get("chatbot", [])
    if not isinstance(chatbot, list):
        chatbot = []
    active_chatbot = _clean_chatbot_messages(chatbot[active_start:])
    return {
        "schema_version": 2,
        "user_id": str(state.get("user_id", "") or ""),
        "page": page,
        "hub_section": hub_section,
        "history_session_id": history_session_id or str(state.get("history_session_id", "") or ""),
        "current_session_id": str(state.get("current_session_id", "") or "") if active_chatbot else "",
        "session_start_time": str(session_start_time or "") if active_chatbot else "",
        "previous_chatbot": _clean_chatbot_messages(chatbot[:active_start]),
        "chatbot": active_chatbot,
        "active_chatbot_start_index": 0,
        "retrieval_records": _clean_retrieval_records(state.get("retrieval_records", [])),
        "l3_records": state.get("l3_records", []) if isinstance(state.get("l3_records", []), list) else [],
    }


def remember_chat_page(state: Dict[str, Any]) -> Dict[str, str]:
    if not isinstance(state, dict) or not state.get("authenticated") or not state.get("user_id"):
        return gr.skip()
    return _browser_session_payload(state, page="chat")


def remember_consult_overview(state: Dict[str, Any]) -> Dict[str, str]:
    return _browser_session_payload(state, page="consult", hub_section="overview")


def remember_consult_report(state: Dict[str, Any]) -> Dict[str, str]:
    return _browser_session_payload(state, page="consult", hub_section="report")


def remember_consult_history(state: Dict[str, Any], history_session_id: str = "") -> Dict[str, str]:
    return _browser_session_payload(state, page="consult", hub_section="history", history_session_id=history_session_id)


def clear_browser_session() -> Dict[str, str]:
    return {}


def restore_browser_session(browser_payload: Dict[str, Any]) -> Tuple[Any, ...]:
    payload = browser_payload if isinstance(browser_payload, dict) else {}
    user_id = str(payload.get("user_id", "") or "").strip()
    profile = find_user_profile(user_id) if user_id else None
    if not profile:
        # The initial page already defaults to the login view. Returning a full
        # "logged out" layout here can race with a quick first login and hide
        # the chat view after login succeeds, so an empty browser session is a
        # true no-op.
        return tuple(gr.skip() for _ in range(34))

    _normalize_user_entry(profile)
    user_config = _user_config_from_profile(profile)
    effective_user_api_key = "" if user_config["config_api_mode"] == "trial" else user_config["config_api_key"]
    state = empty_state()
    state["authenticated"] = True
    state["user_id"] = user_id
    state["display_name"] = str(profile.get("display_name") or user_id)
    phone_number = str(profile.get("phone_number", "") or "")
    phone_verified_at = str(profile.get("phone_verified_at", "") or "")
    state["phone_number"] = phone_number
    state["phone_verified_at"] = phone_verified_at
    state["config"] = {
        "model_backend": user_config["config_model_backend"],
        "agent_backend": user_config["config_agent_backend"],
        "summary_backend": user_config["config_summary_backend"],
        "supervisor_backend": user_config["config_supervisor_backend"],
        "graph_backend": user_config["config_graph_backend"],
        "router_backend": user_config["config_router_backend"],
        "api_mode": user_config["config_api_mode"],
        "api_provider": user_config["config_api_provider"],
        "api_base_url": user_config["config_api_base_url"],
        "api_key": effective_user_api_key,
        "model": user_config["config_model"],
        "config_model_keys": user_config["config_model_keys"],
    }
    state["trial_sessions_used"] = _trial_used_from_profile(profile)
    try:
        payload_schema_version = int(payload.get("schema_version", 0) or 0)
    except (TypeError, ValueError):
        payload_schema_version = 0
    restored_active_chatbot = _clean_chatbot_messages(payload.get("chatbot", []))
    if payload_schema_version < 2 and "previous_chatbot" not in payload:
        # Older BrowserState versions sometimes stored historical messages as
        # active chat. Dropping them here prevents slow reloads and duplicated
        # old sessions; the latest archived session is loaded below instead.
        restored_active_chatbot = []
    if "previous_chatbot" in payload:
        previous_messages = _clean_chatbot_messages(payload.get("previous_chatbot", []))
    else:
        previous_messages = archived_chatbot_messages(user_id)
    restored_chatbot = previous_messages + restored_active_chatbot
    restored_retrieval_records = _clean_retrieval_records(payload.get("retrieval_records", [])) or load_recent_retrieval_records(user_id)
    restored_l3_records = payload.get("l3_records", []) if isinstance(payload.get("l3_records", []), list) else []
    restored_session_id = str(payload.get("current_session_id", "") or "").strip()
    restored_start_time = str(payload.get("session_start_time", "") or "").strip()
    try:
        active_start_index = len(previous_messages)
    except (TypeError, ValueError):
        active_start_index = len(previous_messages)
    active_start_index = min(active_start_index, len(restored_chatbot))
    state["chatbot"] = restored_chatbot
    state["active_chatbot_start_index"] = active_start_index
    state["retrieval_records"] = restored_retrieval_records
    state["l3_records"] = restored_l3_records
    state["current_session_id"] = restored_session_id
    if restored_active_chatbot:
        state = ensure_agent_ready(state)
        agent = state.get("agent")
        if agent is not None:
            agent.session_id = restored_session_id or agent.session_id
            agent.session_messages = _chatbot_to_session_messages(restored_chatbot[active_start_index:])
            state["current_session_id"] = agent.session_id or restored_session_id
            if restored_start_time:
                try:
                    agent.session_start_time = datetime.fromisoformat(restored_start_time)
                except Exception:
                    pass

    choices = session_choices(user_id)
    selected_session = dropdown_value(str(payload.get("history_session_id", "") or ""), choices) or latest_choice_value(choices)
    hub_section = str(payload.get("hub_section", "overview") or "overview")
    if hub_section not in {"overview", "report", "history"}:
        hub_section = "overview"
    state["hub_section"] = hub_section
    if hub_section == "history":
        hub_html, consult_status, _ = render_consultation_hub(user_id, "history", selected_session)
        history_update = dropdown_update(choices, selected_session, bool(choices))
    else:
        hub_html, consult_status, meta = render_consultation_hub(user_id, hub_section, selected_session)
        selected_session = dropdown_value(meta.get("history_session_id", selected_session), choices) or selected_session
        history_update = dropdown_update(choices, selected_session, False)
    state["history_session_id"] = selected_session
    detail_html = meta.get("detail_html", "") if 'meta' in locals() and isinstance(meta, dict) else ""

    page = str(payload.get("page", "chat") or "chat")
    show_consult = page == "consult"
    memory_html = build_session_memory_panel(user_id, state["display_name"])
    browser_update = _browser_session_payload(
        state,
        page="consult" if show_consult else "chat",
        hub_section=hub_section,
        history_session_id=selected_session,
    )
    return (
        state,
        f"已恢复登录：{state['display_name']} ({user_id})",
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=not show_consult),
        state["display_name"],
        memory_html,
        history_update,
        hub_html,
        gr.update(value=detail_html, visible=bool(detail_html) and show_consult),
        gr.update(visible=show_consult),
        "",
        gr.update(value=""),
        gr.update(value=f"我的咨询 · {state['display_name']}"),
        gr.update(value=user_config["config_supervisor_backend"]),
        gr.update(value=user_config["config_graph_backend"]),
        gr.update(value=user_config["config_router_backend"]),
        gr.update(value=user_config["config_agent_backend"]),
        gr.update(value=user_config["config_summary_backend"]),
        gr.update(value=user_config["config_router_backend"]),
        gr.update(value=user_config["config_api_provider"]),
        gr.update(value=effective_user_api_key),
        gr.update(value=user_config["config_model"]),
        gr.update(value=user_config["config_api_mode"]),
        _trial_quota_message(profile),
        gr.update(value=phone_number),
        _form_message(f"当前已绑定：{_masked_phone(phone_number)}。", "ok") if phone_number and phone_verified_at else "",
        render_retrieval_records(state.get("retrieval_records", []), state["display_name"]),
        gr.update(visible=hub_section == "report" and show_consult),
        gr.update(visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=restored_chatbot, visible=True),
        browser_update,
    )


def ensure_agent_ready(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录成功后再进入会话。")
    if state.get("agent_ready") and state.get("agent") is not None:
        return state

    timings: List[Tuple[str, float]] = []

    def mark(label: str, started_at: float) -> None:
        timings.append((label, time.perf_counter() - started_at))

    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    api_runtime = _resolved_api_runtime_config(config)
    primary_backend = str(config.get("model_backend", config.get("agent_backend", "gpt")) or "gpt")
    api_error = _require_api_key_if_needed(
        state,
        primary_backend,
    )
    if api_error:
        raise gr.Error(api_error)
    started = time.perf_counter()
    agent = build_agent(
        primary_backend,
        primary_backend,
        primary_backend,
        primary_backend,
        primary_backend,
        api_key=api_runtime["api_key"],
        base_url=api_runtime["base_url"],
        model=api_runtime["model"],
    )
    mark("build_agent", started)
    started = time.perf_counter()
    session_info = agent.start_session(state["user_id"])
    mark("start_session", started)
    for label, elapsed in session_info.get("preload_timings", []):
        timings.append((f"start.{label}", elapsed))
    started = time.perf_counter()
    state["agent"] = agent
    state["current_session_id"] = session_info.get("session_id", "")
    state["preloaded_session_context"] = {
        "l1_summary": session_info.get("l1_summary"),
        "treatment_report": session_info.get("treatment_report"),
        "preloaded_context_text": session_info.get("preloaded_context_text", ""),
    }
    state["agent_ready"] = True
    state["agent_ready_timings"] = timings
    return state


def charge_trial_session_if_needed(state: Dict[str, Any]) -> Dict[str, Any]:
    config = state.get("config", {}) if isinstance(state.get("config", {}), dict) else {}
    if str(config.get("api_mode", "own") or "own").strip().lower() != "trial":
        return state
    if not _needs_remote_api(str(config.get("model_backend", config.get("agent_backend", "gpt")) or "gpt")):
        return state
    if not _server_trial_api_available():
        raise gr.Error("服务器试用 Key 未配置，暂时不能使用试用模式。")

    session_id = str(state.get("current_session_id", "") or "")
    if session_id and state.get("trial_charged_session_id") == session_id:
        return state

    registry, profile = _find_registry_user(str(state.get("user_id", "") or ""))
    if not profile:
        raise gr.Error("账号信息不存在，请重新登录后再试。")
    used = _trial_used_from_profile(profile)
    if used >= TRIAL_SESSION_LIMIT:
        raise gr.Error(f"试用额度已用完（{TRIAL_SESSION_LIMIT}/{TRIAL_SESSION_LIMIT} 次会话）。请填写自己的 API Key 后继续使用。")
    profile["trial_sessions_used"] = used + 1
    save_user_registry(registry)
    state["trial_sessions_used"] = used + 1
    state["trial_charged_session_id"] = session_id
    return state


def format_agent_ready_timings(state: Dict[str, Any]) -> str:
    timings = state.get("agent_ready_timings", [])
    if not timings:
        return ""
    detail = "，".join(f"{label}={elapsed:.2f}s" for label, elapsed in timings)
    return f"预加载耗时：{detail}"


def _html_escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def render_json_value(value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return '<div class="empty-card">暂无内容</div>'
        rows = []
        for key, item in value.items():
            rows.append(
                '<div class="json-row">'
                f'<div class="json-key">{_html_escape(key)}</div>'
                f'<div class="json-value">{render_json_value(item)}</div>'
                "</div>"
            )
        return "".join(rows)
    if isinstance(value, list):
        if not value:
            return '<div class="empty-card">暂无内容</div>'
        items = "".join(
            f'<li><span class="list-index">{index}</span><div class="list-content">{render_json_value(item)}</div></li>'
            for index, item in enumerate(value, start=1)
        )
        return f'<ol class="json-list">{items}</ol>'
    text = _html_escape(value).replace("\n", "<br>")
    return f'<div class="json-text">{text or "暂无内容"}</div>'


REPORT_FIELD_LABELS = {
    "emotion_trend": "情绪变化轨迹",
    "key_progress": "关键进展",
    "risk_note": "风险评估",
    "treatment_phase": "咨询阶段",
    "next_focus": "下一阶段重点",
}


def localize_report_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {REPORT_FIELD_LABELS.get(str(key), str(key)): localize_report_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [localize_report_payload(item) for item in payload]
    return payload


def render_document_card(title: str, payload: Any, *, subtitle: str = "", path: str = "") -> str:
    subtitle_html = f'<div class="doc-subtitle">{_html_escape(subtitle)}</div>' if subtitle else ""
    return (
        '<section class="doc-card">'
        f'<div class="doc-title">{_html_escape(title)}</div>'
        f"{subtitle_html}"
        f'<div class="doc-body">{render_json_value(payload)}</div>'
        "</section>"
    )


def render_fold_panel(title: str, payload: Any, subtitle: str = "") -> str:
    return (
        '<details class="fold-card">'
        f'<summary>{_html_escape(title)}</summary>'
        '<div class="fold-card-body">'
        f'{render_document_card(title, payload, subtitle=subtitle)}'
        "</div>"
        "</details>"
    )




def normalize_retrieval_evidence(retrieval_meta: Dict[str, Any]) -> Dict[str, Any]:
    retrieval_wrapper = retrieval_meta.get("retrieval", {}) if isinstance(retrieval_meta, dict) else {}
    if not isinstance(retrieval_wrapper, dict):
        retrieval_wrapper = {}

    nested_retrieval = retrieval_wrapper.get("retrieval", {})
    retrieval_payload = nested_retrieval if isinstance(nested_retrieval, dict) else retrieval_wrapper
    context_text = retrieval_wrapper.get("context_text", "") if isinstance(retrieval_wrapper.get("context_text", ""), str) else ""

    l3_fragments = retrieval_payload.get("l3_fragments", [])
    graph_relations = retrieval_payload.get("graph_relations", [])
    ranked_evidence = retrieval_payload.get("ranked_evidence", [])
    return {
        "context_text": context_text,
        "l3_fragments": l3_fragments if isinstance(l3_fragments, list) else [],
        "graph_relations": graph_relations if isinstance(graph_relations, list) else [],
        "ranked_evidence": ranked_evidence if isinstance(ranked_evidence, list) else [],
    }

def _compact_relation_text(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    source = str(item.get("source", "") or "").strip()
    relation = str(item.get("relationship", item.get("relation", "")) or "").strip()
    target = str(item.get("target", item.get("destination", "")) or "").strip()
    if source or target:
        if relation and target:
            return f"{source} → {target}（{relation}）"
        if relation:
            return f"{source}（{relation}）"
        return f"{source} → {target}".strip(" →")
    memory = str(item.get("memory", "") or "").strip()
    return memory


def _compact_l3_text(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    timestamp = str(item.get("timestamp", "") or "")[:10]
    memory = str(item.get("memory", "") or "").strip()
    if timestamp and memory:
        return f"[{timestamp}] {memory}"
    return memory or timestamp


def _compact_retrieval_evidence_text(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    memory_type = str(item.get("memory_type", "") or "").strip()
    if memory_type == "graph_relation" or item.get("source") or item.get("target"):
        return _compact_relation_text(item)
    return _compact_l3_text(item)


def _combined_retrieval_evidence_items(evidence: Dict[str, Any], limit: int = 6) -> List[str]:
    if not isinstance(evidence, dict):
        return []

    raw_items = evidence.get("ranked_evidence", [])
    if not isinstance(raw_items, list) or not raw_items:
        raw_items = []
        l3_fragments = evidence.get("l3_fragments", [])
        graph_relations = evidence.get("graph_relations", [])
        if isinstance(l3_fragments, list):
            raw_items.extend(l3_fragments)
        if isinstance(graph_relations, list):
            raw_items.extend(graph_relations)

    items: List[str] = []
    seen = set()
    for item in raw_items:
        text = _compact_retrieval_evidence_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
        if len(items) >= limit:
            break
    return items


def render_retrieval_records(records: List[Dict[str, Any]], display_name: str = "") -> str:
    if not records:
        return render_document_card("最近五轮检索证据", {"提示": "尚无检索记录。"}, subtitle=display_name)

    cards: List[str] = []
    recent_records = list(reversed(_clean_retrieval_records(records)))
    for recent_index, record in enumerate(recent_records, start=1):
        if not isinstance(record, dict):
            continue
        route = record.get("route", {}) if isinstance(record.get("route", {}), dict) else {}
        evidence = record.get("evidence", {}) if isinstance(record.get("evidence", {}), dict) else {}
        evidence_items = _combined_retrieval_evidence_items(evidence, limit=6)
        payload = {
            "路由判决": {
                "是否检索": "\u662f" if bool(route.get("need_retrieval", False)) else "\u5426",
                "置信度": route.get("confidence", ""),
                "原因": route.get("reason", ""),
                "检索焦点": route.get("retrieval_focus", ""),
            },
            "索引证据": evidence_items or ["暂无内容"],
        }
        recent_label = _chinese_ordinal(recent_index)
        title = f"最近{recent_label}轮索引"
        original_turn = record.get("turn_index", "")
        subtitle_parts = []
        if original_turn:
            subtitle_parts.append(f"原始第{original_turn}轮")
        if record.get("created_at", ""):
            subtitle_parts.append(str(record.get("created_at", "")))
        subtitle = " · ".join(subtitle_parts)
        cards.append(render_fold_panel(title, payload, subtitle=subtitle))
    return '<div class="fold-grid">' + "".join(cards) + "</div>"


def build_session_memory_panel(user_id: str, display_name: str) -> str:
    profile = read_json_payload(whole_summary_path(user_id), {})
    report_payload, _ = latest_treatment_report_payload(user_id)
    summary_payload, _ = latest_session_summary_payload(user_id)
    return (
        '<div class="fold-grid">'
        f'{render_fold_panel("我的画像", profile, subtitle=display_name)}'
        f'{render_fold_panel("最新会话总结", summary_payload, subtitle=display_name)}'
        f'{render_fold_panel("最新督导师报告", localize_report_payload(report_payload), subtitle=display_name)}'
        '</div>'
    )


def _chinese_ordinal(index: int) -> str:
    digits = "零一二三四五六七八九"
    if index <= 0:
        return str(index)
    if index < 10:
        return digits[index]
    if index == 10:
        return "十"
    if index < 20:
        return "十" + digits[index % 10]
    if index < 100:
        ten, one = divmod(index, 10)
        return digits[ten] + "十" + (digits[one] if one else "")
    return str(index)


def _session_title(session: Dict[str, Any]) -> str:
    title = str(session.get("session_title", "") or "").strip()
    if title and bool(session.get("session_title_custom", False)):
        return title
    session_id = str(session.get("session_id", "") or "").strip()
    match = re.search(r"(\d+)", session_id)
    if match:
        return f"第{_chinese_ordinal(int(match.group(1)))}场"
    return session_id or "未命名会话"


def session_choices(user_id: str) -> List[Tuple[str, str]]:
    doc = load_user_sessions(user_id)
    sessions = doc.get("sessions", []) if isinstance(doc.get("sessions", []), list) else []
    choices = [
        (f"{_session_title(item)}（{item.get('session_id')}）", str(item.get("session_id")))
        for item in sessions
        if isinstance(item, dict) and item.get("session_id")
    ]
    return choices


def session_title_for_id(user_id: str, session_id: str) -> str:
    session_id = normalize_session_id_choice(session_id)
    doc = load_user_sessions(user_id)
    sessions = doc.get("sessions", []) if isinstance(doc.get("sessions", []), list) else []
    for item in sessions:
        if isinstance(item, dict) and str(item.get("session_id", "")) == str(session_id):
            return _session_title(item)
    return ""


def normalize_session_id_choice(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"Session\d+", text):
        return text
    match = re.search(r"(Session\d+)", text)
    if match:
        return match.group(1)
    return text


def render_history_session_detail(user_id: str, session_id: str) -> str:
    doc = load_user_sessions(user_id)
    sessions = doc.get("sessions", []) if isinstance(doc.get("sessions", []), list) else []
    selected = None
    for item in sessions:
        if isinstance(item, dict) and str(item.get("session_id", "")) == str(session_id):
            selected = item
            break
    if not selected:
        return render_document_card("历史会话", {"提示": "请选择一个会话编号。"})

    turns = []
    for turn in selected.get("turns", []) or []:
        if isinstance(turn, dict):
            turns.append({"用户": turn.get("user", ""), "咨询师": turn.get("assistant", "")})
    body = {
        "会话名称": _session_title(selected),
        "会话编号": selected.get("session_id", ""),
        "归档时间": selected.get("archived_at", ""),
        "对话轮数": selected.get("turn_count", len(turns)),
        "历史交互": turns,
    }
    return render_document_card("历史会话", body, subtitle=f"{_session_title(selected)} · {selected.get('session_id', '')}")


def render_consultation_hub(user_id: str, section: str = "overview", session_id: str = "") -> Tuple[str, str, Dict[str, Any]]:
    choices = session_choices(user_id)
    latest_history = session_id or latest_choice_value(choices)
    if section == "profile":
        profile_html = render_document_card("我的画像", read_json_payload(whole_summary_path(user_id), {}))
        return profile_html, "已打开长期记忆。", {"history_visible": False, "history_session_id": latest_history, "detail_html": ""}
    if section == "report":
        summary_doc = load_user_session_summaries(user_id)
        full_report_payload = {
            "final_l1_summary": read_json_payload(whole_summary_path(user_id), {}),
            "session_results": collect_archived_session_results(user_id),
            "all_l2_summaries": summary_doc.get("sessions", []) if isinstance(summary_doc, dict) else [],
        }
        profile_section, detail_section = render_full_report_sections(user_id, full_report_payload)
        return profile_section, "已打开阶段摘要。", {
            "history_visible": False,
            "history_session_id": latest_history,
            "detail_html": detail_section,
        }
    if section == "history":
        return render_history_session_detail(user_id, latest_history), "已打开报告追踪。", {
            "history_visible": True,
            "history_session_id": latest_history,
            "detail_html": "",
        }
    return (
        render_document_card(
            "我的咨询",
            {
                "连续咨询": "返回聊天界面继续对话",
                "长期记忆": "查看我的画像",
                "阶段摘要": "查看完整报告",
                "报告追踪": "查看历史会话",
            },
        ),
        "已打开我的咨询。",
        {"history_visible": False, "history_session_id": latest_history, "detail_html": ""},
    )


def read_json_payload(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def resource_button_updates(active: str) -> Tuple[dict, dict, dict, dict]:
    names = ("profile", "supervisor", "report", "history")
    return tuple(gr.update(variant="primary" if name == active else "secondary") for name in names)


def latest_session_summary_payload(user_id: str) -> Tuple[Any, str]:
    path = sessions_summaries_path(user_id)
    doc = load_user_session_summaries(user_id)
    sessions = doc.get("sessions", []) if isinstance(doc.get("sessions", []), list) else []
    if not sessions:
        return {}, str(path.resolve())
    latest = max(
        (item for item in sessions if isinstance(item, dict)),
        key=lambda item: (item.get("id", 0), str(item.get("archived_at", ""))),
        default={},
    )
    return latest.get("summary", latest), str(path.resolve())


def latest_treatment_report_payload(user_id: str) -> Tuple[Any, str]:
    path = trends_path(user_id)
    doc = load_user_trends(user_id)
    reports = doc.get("reports", []) if isinstance(doc.get("reports", []), list) else []
    if not reports:
        return {}, str(path.resolve())
    latest = max(
        (item for item in reports if isinstance(item, dict)),
        key=lambda item: (item.get("id", 0), str(item.get("generated_at", ""))),
        default={},
    )
    return latest.get("report", latest), str(path.resolve())


def latest_consulting_report_html(user_id: str) -> Tuple[str, str]:
    safe_id = safe_user_id(user_id)
    candidates = sorted(
        CONSULTING_REPORTS_DIR.glob(f"{safe_id}_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return "", ""
    path = candidates[0]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "", str(path.resolve())
    return render_current_full_report(user_id, payload, str(path.resolve())), str(path.resolve())


def normalize_summary_text(summary: Any) -> Dict[str, Any]:
    if isinstance(summary, dict):
        return summary
    text = str(summary or "").strip()
    if not text:
        return {}
    return {"摘要": text}


def session_report_cards(user_id: str) -> str:
    summaries_doc = load_user_session_summaries(user_id)
    trends_doc = load_user_trends(user_id)
    summaries = summaries_doc.get("sessions", []) if isinstance(summaries_doc.get("sessions", []), list) else []
    reports = trends_doc.get("reports", []) if isinstance(trends_doc.get("reports", []), list) else []
    report_by_session = {
        str(item.get("session_id", "")): item.get("report", {})
        for item in reports
        if isinstance(item, dict)
    }
    cards: List[str] = []
    for item in summaries:
        if not isinstance(item, dict):
            continue
        session_id = str(item.get("session_id", ""))
        summary = normalize_summary_text(item.get("summary"))
        report = localize_report_payload(report_by_session.get(session_id, {}))
        cards.append(
            '<section class="doc-card session-report-card">'
            f'<div class="doc-title">{_html_escape(session_id or "Session")}</div>'
            f'<div class="doc-subtitle">记忆落库时间：{_html_escape(item.get("archived_at", ""))}</div>'
            '<div class="two-report-cols">'
            f'<div>{render_document_card("会话记忆摘要", summary)}</div>'
            f'<div>{render_document_card("督导师报告", report)}</div>'
            '</div>'
            '</section>'
        )
    if not cards:
        return render_document_card("Session 记忆与督导师报告", {})
    return "".join(cards)


def render_current_full_report(user_id: str, payload: Dict[str, Any], path: str) -> str:
    l1 = payload.get("final_l1_summary") or {}
    sessions = payload.get("session_results", []) if isinstance(payload.get("session_results", []), list) else []
    l2 = payload.get("all_l2_summaries", []) if isinstance(payload.get("all_l2_summaries", []), list) else []
    overview = {
        "用户": user_id,
        "已归档会话数": len(sessions),
        "会话摘要数": len(l2),
    }
    return (
        '<div class="full-report-shell">'
        f'{render_document_card("完整报告概览", overview)}'
        f'{render_document_card("长期画像", l1)}'
        f'{session_report_cards(user_id)}'
        '</div>'
    )


def render_full_report_sections(user_id: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    l1 = payload.get("final_l1_summary") or {}
    sessions = payload.get("session_results", []) if isinstance(payload.get("session_results", []), list) else []
    l2 = payload.get("all_l2_summaries", []) if isinstance(payload.get("all_l2_summaries", []), list) else []
    overview = {
        "用户": user_id,
        "已归档会话数": len(sessions),
        "会话摘要数": len(l2),
    }
    overview_html = render_document_card("完整报告概览", overview)
    detail_html = (
        '<div class="full-report-shell">'
        f'{render_document_card("长期画像", l1)}'
        f'{session_report_cards(user_id)}'
        '</div>'
    )
    return overview_html, detail_html


def send_message(user_input: str, state: Dict[str, Any]):
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录成功后再进入会话。")
    state, _, _, _ = refresh_background_finalize_state(state)
    retrieval_panel_html = render_retrieval_records(state.get("retrieval_records", []), state.get("display_name", ""))
    if not user_input.strip():
        yield "", state, state.get("chatbot", []), "请输入消息后再发送。", retrieval_panel_html, gr.skip()
        return

    state = ensure_agent_ready(state)
    state = charge_trial_session_if_needed(state)
    agent: EmpathyAgent = state["agent"]
    chatbot = list(state.get("chatbot", []))
    chatbot.append({"role": "user", "content": user_input.strip()})
    chatbot.append({"role": "assistant", "content": ""})
    state["chatbot"] = chatbot
    yield "", state, chatbot, f"正在生成回复：{state.get('current_session_id', '')}", retrieval_panel_html, gr.skip()

    result: Dict[str, Any] = {}
    for chunk in agent.generate_response_stream(user_input.strip()):
        if not isinstance(chunk, dict):
            continue
        result = chunk
        chatbot[-1]["content"] = str(chunk.get("response", chatbot[-1]["content"]) or "")
        state["chatbot"] = chatbot
        if not chunk.get("done"):
            yield "", state, chatbot, f"正在生成回复：{state.get('current_session_id', '')}", retrieval_panel_html, gr.skip()

    response_text = str(result.get("response", "") or "").strip()
    l3_records = list(state.get("l3_records", []))
    l3_records.append(
        {
            "turn_index": len(l3_records) + 1,
            "user": user_input.strip(),
            "assistant": response_text,
            "l3_memory_result": result.get("l3_memory_result", {}),
            "retrieval_context": result.get("retrieval_context", ""),
        }
    )
    state["l3_records"] = l3_records

    retrieval_meta = result.get("retrieval_meta", {}) if isinstance(result.get("retrieval_meta", {}), dict) else {}
    retrieval = normalize_retrieval_evidence(retrieval_meta)
    retrieval_records = list(state.get("retrieval_records", []))
    retrieval_records.append(
        {
            "turn_index": len(retrieval_records) + 1,
            "created_at": datetime.now().isoformat(),
            "user": user_input.strip(),
            "assistant": response_text,
            "route": {
                "need_retrieval": bool(retrieval_meta.get("should_retrieve", False)),
                "confidence": (retrieval_meta.get("decision", {}) or {}).get("confidence", ""),
                "reason": (retrieval_meta.get("decision", {}) or {}).get("reason", ""),
                "retrieval_focus": (retrieval_meta.get("decision", {}) or {}).get("retrieval_focus", ""),
            },
            "retrieval_query": retrieval_meta.get("retrieval_query", ""),
              "evidence": {
                "context_text": retrieval.get("context_text", ""),
                "l3_fragments": retrieval.get("l3_fragments", []),
                "graph_relations": retrieval.get("graph_relations", []),
                "ranked_evidence": retrieval.get("ranked_evidence", []),
              },
        }
    )
    state["retrieval_records"] = _clean_retrieval_records(retrieval_records)
    save_recent_retrieval_records(state.get("user_id", ""), state["retrieval_records"])
    retrieval_panel_html = render_retrieval_records(state["retrieval_records"], state.get("display_name", ""))

    timing_text = format_agent_ready_timings(state)
    status = f"已回复。当前会话：{state.get('current_session_id', '')}"
    if timing_text:
        status = f"{status}\n{timing_text}"
    if str((state.get("config", {}) or {}).get("api_mode", "own")) == "trial":
        status = f"{status}\n{_trial_quota_text(state)}"
    yield "", state, chatbot, status, retrieval_panel_html, _browser_session_payload(state)


def collect_archived_session_results(user_id: str) -> List[Dict[str, Any]]:
    session_results: List[Dict[str, Any]] = []
    payload = load_user_sessions(user_id)
    for session in payload.get("sessions", []):
        chat_pairs = [
            {"user": turn.get("user", ""), "assistant": turn.get("assistant", "")}
            for turn in session.get("turns", [])
        ]
        session_results.append(
            {
                "session_label": session.get("session_id", ""),
                "run_session_id": session.get("run_session_id", session.get("session_id", "")),
                "chat_pairs": chat_pairs,
                "l3_records": [],
                "end_result": {},
            }
        )
    return session_results


def finalize_session_if_needed(state: Dict[str, Any], *, restart_session: bool = True) -> Dict[str, Any]:
    state = ensure_agent_ready(state)
    agent: EmpathyAgent = state["agent"]
    active_messages = list(getattr(agent, "session_messages", []) or [])
    active_chatbot = _active_chatbot_messages(state)
    if not active_messages and not active_chatbot:
        return state

    end_result = agent.end_session()

    chat_pairs = []
    if not active_chatbot:
        active_chatbot = _session_messages_to_chatbot(active_messages)
    for idx in range(0, len(active_chatbot), 2):
        user_msg = active_chatbot[idx]
        assistant_msg = active_chatbot[idx + 1] if idx + 1 < len(active_chatbot) else {"content": ""}
        if user_msg.get("role") == "user":
            chat_pairs.append({"user": user_msg.get("content", ""), "assistant": assistant_msg.get("content", "")})
    output_paths = record_session_outputs(
        user_id=state["user_id"],
        display_name=state["display_name"],
        run_session_id=state["current_session_id"],
        chat_pairs=chat_pairs,
        l2_summary=end_result.get("l2_summary", {}),
        treatment_report=end_result.get("treatment_report", {}),
        l1_summary=end_result.get("l1_summary", {}),
        archived_at=datetime.now().isoformat(),
    )
    state["last_output_paths"] = output_paths

    state["chatbot"] = active_chatbot
    state["active_chatbot_start_index"] = len(active_chatbot)
    state["l3_records"] = []
    if restart_session:
        session_info = agent.start_session(state["user_id"])
        state["current_session_id"] = session_info.get("session_id", "")
        state["preloaded_session_context"] = {
            "l1_summary": session_info.get("l1_summary"),
            "treatment_report": session_info.get("treatment_report"),
            "preloaded_context_text": session_info.get("preloaded_context_text", ""),
        }
    else:
        state["current_session_id"] = ""
        state["preloaded_session_context"] = {}
        state["agent_ready"] = False
        state["agent"] = None
    return state


def ensure_report_bundle(state: Dict[str, Any], prefix: str = "manual") -> Dict[str, Any]:
    state = finalize_session_if_needed(state, restart_session=True)
    state = ensure_agent_ready(state)
    agent: EmpathyAgent = state["agent"]
    archived_session_results = collect_archived_session_results(state["user_id"])
    summary_doc = load_user_session_summaries(state["user_id"])
    trends_doc = load_user_trends(state["user_id"])
    trend_reports = trends_doc.get("reports", []) if isinstance(trends_doc.get("reports", []), list) else []
    latest_report = {}
    if trend_reports:
        latest_report = max(
            trend_reports,
            key=lambda item: str(item.get("generated_at", "")) if isinstance(item, dict) else "",
        ).get("report", {})
    bundle = build_result_bundle(
        agent=agent,
        user_id=state["user_id"],
        session_results=archived_session_results,
        completed_sessions=[item.get("session_label", "") for item in archived_session_results],
        status="active",
        include_retrieval_examples=False,
    )
    bundle["all_l2_summaries"] = summary_doc.get("sessions", [])
    bundle["latest_treatment_report"] = latest_report
    result_path = consulting_report_json_path(state["user_id"], prefix=prefix)
    result_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    html_text = render_current_full_report(state["user_id"], bundle, str(result_path.resolve()))
    html_path = consulting_report_html_path(state["user_id"], prefix=prefix)
    html_path.write_text(html_text, encoding="utf-8")
    bundle["_html_text"] = html_text
    bundle["_html_path"] = str(html_path.resolve())
    return bundle


def show_profile(state: Dict[str, Any]) -> Tuple[str, str, dict, dict, dict, dict]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    payload = read_json_payload(whole_summary_path(state["user_id"]), {})
    html_text = render_document_card(
        "我的画像",
        payload,
        subtitle=state["display_name"],
    )
    return html_text, "已加载我的画像。", *resource_button_updates("profile")


def show_supervisor_report(state: Dict[str, Any]) -> Tuple[str, str, dict, dict, dict, dict]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    payload, _ = latest_treatment_report_payload(state["user_id"])
    html_text = render_document_card("最新督导师报告", localize_report_payload(payload), subtitle=state["display_name"])
    return html_text, "已加载最新督导师报告。", *resource_button_updates("supervisor")


def show_full_report(state: Dict[str, Any]) -> Tuple[str, str, dict, dict, dict, dict]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    html_text, _ = latest_consulting_report_html(state["user_id"])
    if html_text:
        return html_text, "已加载完整报告。", *resource_button_updates("report")
    summary_doc = load_user_session_summaries(state["user_id"])
    virtual_payload = {
        "final_l1_summary": read_json_payload(whole_summary_path(state["user_id"]), {}),
        "session_results": collect_archived_session_results(state["user_id"]),
        "all_l2_summaries": summary_doc.get("sessions", []) if isinstance(summary_doc, dict) else [],
    }
    html_text = render_current_full_report(state["user_id"], virtual_payload, "")
    return html_text, "已加载完整报告。", *resource_button_updates("report")


def _export_html_document(title: str, body_html: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="zh-CN">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_html_escape(title)}</title>\n"
        "<style>\n"
        f"{APP_CSS}\n"
        "body { margin: 0; padding: 24px; }\n"
        ".export-shell { max-width: 1180px; margin: 0 auto; }\n"
        "</style>\n"
        "</head>\n"
        '<body class="gradio-container">\n'
        '<main class="export-shell">\n'
        f"{body_html}\n"
        "</main>\n"
        "</body>\n"
        "</html>\n"
    )


def build_export_report_html(state: Dict[str, Any]) -> Path:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录后再导出报告。")

    user_id = state["user_id"]
    summary_doc = load_user_session_summaries(user_id)
    payload = {
        "final_l1_summary": read_json_payload(whole_summary_path(user_id), {}),
        "session_results": collect_archived_session_results(user_id),
        "all_l2_summaries": summary_doc.get("sessions", []) if isinstance(summary_doc, dict) else [],
    }
    body_html = render_current_full_report(user_id, payload, "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = consulting_report_html_path(user_id, prefix=f"export_{timestamp}")
    html_path.write_text(
        _export_html_document(f"MemAgent咨询报告_{safe_user_id(user_id)}", body_html),
        encoding="utf-8",
    )
    return html_path


def open_html_export_confirm(state: Dict[str, Any]) -> Tuple[dict, str, dict]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录后再导出报告。")
    return (
        gr.update(visible=True),
        _form_message("确认导出当前用户的完整 HTML 报告吗？导出不会结束当前会话。", "ok"),
        gr.update(value=None, visible=False),
    )


def cancel_html_export() -> Tuple[dict, str, dict]:
    return gr.update(visible=False), _form_message("已取消导出。"), gr.update(value=None, visible=False)


def confirm_html_export(state: Dict[str, Any]) -> Tuple[dict, str, dict]:
    html_path = build_export_report_html(state)
    return (
        gr.update(visible=True),
        _form_message(f"HTML 报告已生成：{html_path.name}", "ok"),
        gr.update(value=str(html_path.resolve()), visible=True),
    )


def render_history_sessions(user_id: str) -> str:
    doc = load_user_sessions(user_id)
    sessions = doc.get("sessions", []) if isinstance(doc.get("sessions", []), list) else []
    if not sessions:
        return render_document_card("历史会话", {"提示": "暂无历史会话。"})

    cards: List[str] = []
    for index, session in enumerate(sessions, start=1):
        if not isinstance(session, dict):
            continue
        turns = []
        for turn in session.get("turns", []) or []:
            if not isinstance(turn, dict):
                continue
            turns.append(
                {
                    "用户": turn.get("user", ""),
                    "咨询师": turn.get("assistant", ""),
                }
            )
        body = {
            "会话名称": _session_title(session),
            "会话编号": session.get("session_id") or f"Session{index}",
            "归档时间": session.get("archived_at", ""),
            "对话轮数": session.get("turn_count", len(turns)),
            "历史交互": turns,
        }
        cards.append(render_document_card(f"历史会话 {index}", body, subtitle=_session_title(session)))
    return '<div class="history-shell">' + "".join(cards) + "</div>"


def show_history_sessions(state: Dict[str, Any]) -> Tuple[str, str, dict, dict, dict, dict]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    return render_history_sessions(state["user_id"]), "已加载历史会话。", *resource_button_updates("history")


def open_chat_page(state: Dict[str, Any]) -> Tuple[dict, dict, str]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    state, _, _, _ = refresh_background_finalize_state(state)
    return gr.update(visible=True), gr.update(visible=False), "已返回咨询对话。"


def activate_chat_view(state: Dict[str, Any]) -> Tuple[dict, dict, Any]:
    if not isinstance(state, dict) or not state.get("authenticated") or not state.get("user_id"):
        return gr.skip(), gr.skip(), gr.skip()
    return gr.update(visible=True), gr.update(visible=False), gr.update()


def open_consultation_hub(state: Dict[str, Any]) -> Tuple[dict, dict, str, dict, dict, str, dict, dict, dict, dict, dict]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    state, _, _, _ = refresh_background_finalize_state(state)
    choices = session_choices(state["user_id"])
    selected = dropdown_value(state.get("history_session_id", ""), choices) or latest_choice_value(choices)
    html_text, status, meta = render_consultation_hub(state["user_id"], "overview", selected)
    state["hub_section"] = "overview"
    state["history_session_id"] = dropdown_value(meta.get("history_session_id", selected), choices) or ""
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        html_text,
        gr.update(value=meta.get("detail_html", ""), visible=bool(meta.get("detail_html"))),
        dropdown_update(choices, state["history_session_id"], False),
        status,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=None, visible=False),
        gr.update(value="", visible=False),
        gr.update(visible=False),
    )


def show_hub_profile(state: Dict[str, Any]) -> Tuple[str, dict, str, dict]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    state, _, _, _ = refresh_background_finalize_state(state)
    html_text, status, _ = render_consultation_hub(state["user_id"], "profile", state.get("history_session_id", ""))
    return html_text, gr.update(visible=False), status, gr.update(value="", visible=False)


def show_hub_full_report(state: Dict[str, Any]) -> Tuple[str, dict, str, dict, dict, dict, dict, dict, dict]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    state, _, _, _ = refresh_background_finalize_state(state)
    html_text, status, meta = render_consultation_hub(state["user_id"], "report", state.get("history_session_id", ""))
    state["hub_section"] = "report"
    return (
        html_text,
        gr.update(visible=False),
        status,
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=meta.get("detail_html", ""), visible=True),
        gr.update(value="", visible=False),
        gr.update(visible=False),
    )


def show_hub_history(state: Dict[str, Any]) -> Tuple[Dict[str, Any], str, dict, str, dict, dict, dict, dict, dict, dict, dict]:
    if not state.get("authenticated") or not state.get("user_id"):
        raise gr.Error("请先登录。")
    state, _, _, _ = refresh_background_finalize_state(state)
    choices = session_choices(state["user_id"])
    selected = dropdown_value(state.get("history_session_id", ""), choices) or latest_choice_value(choices)
    state["history_session_id"] = dropdown_value(selected, choices) or ""
    html_text, status, _ = render_consultation_hub(state["user_id"], "history", selected)
    state["hub_section"] = "history"
    return (
        state,
        html_text,
        dropdown_update(choices, selected, bool(choices)),
        status,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=None, visible=False),
        gr.update(value="", visible=False),
        gr.update(value=session_title_for_id(state["user_id"], selected), visible=bool(selected)),
        gr.update(visible=bool(selected)),
        gr.update(visible=bool(selected)),
    )


def select_history_session(session_id: str, state: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str, dict, dict, dict, dict, dict, dict, dict]:
    if not state.get("authenticated") or not state.get("user_id"):
        return state, "", "", gr.update(visible=False), gr.update(visible=False), gr.update(value=None, visible=False), gr.update(value="", visible=False), gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False)
    state, _, _, _ = refresh_background_finalize_state(state)
    state["history_session_id"] = normalize_session_id_choice(session_id)
    state["hub_section"] = "history"
    return (
        state,
        render_history_session_detail(state["user_id"], state["history_session_id"]),
        "已切换历史会话。",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=None, visible=False),
        gr.update(value="", visible=False),
        gr.update(value=session_title_for_id(state["user_id"], state["history_session_id"]), visible=bool(state["history_session_id"])),
        gr.update(visible=bool(state["history_session_id"])),
        gr.update(visible=bool(state["history_session_id"])),
    )


def save_history_session_title(session_id: str, title: str, state: Dict[str, Any]) -> Tuple[Dict[str, Any], dict, str, str, dict]:
    if not state.get("authenticated") or not state.get("user_id"):
        return state, gr.update(), "", "请先登录。", gr.update()
    session_id = normalize_session_id_choice(session_id or state.get("history_session_id", ""))
    title = str(title or "").strip()
    if not session_id:
        return state, gr.update(), "", "请先选择一个历史会话。", gr.update()
    if not title:
        return state, gr.update(), "", "会话名称不能为空。", gr.update()
    ok = rename_user_session(state["user_id"], session_id, title)
    if not ok:
        return state, gr.update(), render_history_session_detail(state["user_id"], session_id), "未找到该会话，名称未保存。", gr.update()
    choices = session_choices(state["user_id"])
    state["history_session_id"] = session_id
    return (
        state,
        dropdown_update(choices, session_id, True),
        render_history_session_detail(state["user_id"], session_id),
        "会话名称已保存。",
        gr.update(value=title, visible=True),
    )


def end_session(state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]], str, str, str, Dict[str, Any]]:
    agent: Optional[EmpathyAgent] = state.get("agent")
    active_messages = list(getattr(agent, "session_messages", []) or []) if agent is not None else []
    if not active_messages:
        memory_html = build_session_memory_panel(state["user_id"], state["display_name"]) if state.get("user_id") else ""
        retrieval_html = render_retrieval_records(state.get("retrieval_records", []), state.get("display_name", ""))
        return state, state.get("chatbot", []), "当前没有新的对话内容，本次不会写入记忆或会话记录。", memory_html, retrieval_html, _browser_session_payload(state)
    agent = state["agent"]
    latest_session_chatbot = _active_chatbot_messages(state)
    if not latest_session_chatbot:
        latest_session_chatbot = _session_messages_to_chatbot(active_messages)
    job_id = _schedule_background_finalize(state, agent)
    state["background_finalize_message"] = f"上一会话已进入后台归档：{state['current_session_id']}"

    state["chatbot"] = latest_session_chatbot
    state["active_chatbot_start_index"] = len(latest_session_chatbot)
    state["l3_records"] = []
    state["retrieval_records"] = _clean_retrieval_records(state.get("retrieval_records", []))
    save_recent_retrieval_records(state.get("user_id", ""), state["retrieval_records"])
    agent.session_messages = []
    agent.current_user_id = None
    agent.session_id = None
    agent.session_start_time = None
    agent.preloaded_session_context = {}

    session_info = agent.start_session(state["user_id"])
    state["current_session_id"] = session_info.get("session_id", "")
    state["preloaded_session_context"] = {
        "l1_summary": session_info.get("l1_summary"),
        "treatment_report": session_info.get("treatment_report"),
        "preloaded_context_text": session_info.get("preloaded_context_text", ""),
    }

    memory_html = build_session_memory_panel(state["user_id"], state["display_name"])
    retrieval_html = render_retrieval_records(state.get("retrieval_records", []), state.get("display_name", ""))
    return (
        state,
        state.get("chatbot", []),
        f"当前会话已结束，新会话已开启。后台归档任务：{job_id}",
        memory_html,
        retrieval_html,
        _browser_session_payload(state),
    )


def start_background_memory_warmup() -> None:
    global MEMORY_WARMUP_FUTURE

    enabled = os.getenv("MEMAGENT_BACKGROUND_MEMORY_WARMUP", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return
    if os.getenv("MEMAGENT_ENABLE_MEM0_RUNTIME", "0").strip().lower() not in {"1", "true", "yes"}:
        return

    with MEMORY_WARMUP_LOCK:
        if MEMORY_WARMUP_FUTURE is not None and not MEMORY_WARMUP_FUTURE.done():
            return

        def _run() -> None:
            try:
                from memory.mem0_adapter import Mem0Adapter

                Mem0Adapter()
                print("[MemAgent] 记忆检索模块已完成后台初始化。", flush=True)
            except Exception as exc:
                print(f"[MemAgent] 记忆检索模块后台初始化失败：{type(exc).__name__}: {exc}", flush=True)

        MEMORY_WARMUP_FUTURE = MEMORY_WARMUP_EXECUTOR.submit(_run)


def warmup_ui_callback() -> None:
    start_background_memory_warmup()
    return None


APP_CSS = """
:root {
  --blue-950: #0b2f66;
  --blue-800: #1456c9;
  --blue-700: #1677ff;
  --blue-500: #29a8ff;
  --blue-300: #7bd8ff;
  --blue-100: #eaf6ff;
  --sky-50: #f6fbff;
  --ink: #13233a;
  --muted: #607289;
  --line: #d7e7f7;
  --panel: rgba(255, 255, 255, 0.96);
  --shadow: 0 18px 42px rgba(32, 96, 170, 0.12);
}

.gradio-container {
  min-height: 100vh;
  color: var(--ink);
  background:
    linear-gradient(90deg, rgba(22, 119, 255, 0.035) 1px, transparent 1px),
    linear-gradient(0deg, rgba(22, 119, 255, 0.035) 1px, transparent 1px),
    linear-gradient(180deg, #fbfdff 0%, #f4faff 48%, #ffffff 100%);
  background-size: 32px 32px, 32px 32px, 100% 100%;
  animation: workspaceGrid 34s linear infinite;
}

.gradio-container::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  opacity: 0.9;
  background:
    linear-gradient(180deg, rgba(22, 119, 255, 0.08), transparent 18%),
    linear-gradient(90deg, rgba(41, 168, 255, 0.12), transparent 28%, transparent 72%, rgba(41, 168, 255, 0.08));
}

.gradio-container .main {
  background: transparent;
  padding-top: 18px;
}

.app-shell,
.auth-shell,
.chat-shell {
  max-width: 1180px;
  margin: 0 auto;
}

.hero-layout {
  display: grid !important;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 18px;
  margin: 4px auto 14px;
  padding: 24px 28px;
  border: 1px solid rgba(255, 255, 255, 0.72);
  border-radius: 8px;
  overflow: hidden;
  color: #ffffff;
  background:
    linear-gradient(120deg, rgba(255, 255, 255, 0.14), transparent 34%, rgba(255, 255, 255, 0.08) 68%, transparent 100%),
    linear-gradient(135deg, #1167ed 0%, #1677ff 42%, #22b7ff 100%);
  background-size: 220% 100%, 100% 100%;
  animation: blueSheen 12s ease-in-out infinite;
  box-shadow: 0 18px 48px rgba(22, 119, 255, 0.2);
  position: relative;
}

.hero-layout::before,
.hero-layout::after {
  content: "";
  position: absolute;
  pointer-events: none;
}

.hero-layout::before {
  width: 360px;
  height: 120px;
  right: -96px;
  top: 18px;
  opacity: 0.22;
  background:
    linear-gradient(90deg, rgba(255, 255, 255, 0.82) 0 2px, transparent 2px 22px),
    linear-gradient(0deg, rgba(255, 255, 255, 0.64) 0 2px, transparent 2px 22px);
  background-size: 22px 22px;
  transform: rotate(-10deg);
  animation: blueprintMove 16s linear infinite;
}

.hero-layout::after {
  width: 190px;
  height: 190px;
  right: 58px;
  top: -76px;
  border-radius: 50%;
  opacity: 0.18;
  background:
    radial-gradient(circle, transparent 46%, rgba(255, 255, 255, 0.95) 47% 49%, transparent 50% 100%),
    conic-gradient(from 0deg, rgba(255,255,255,0), rgba(255,255,255,0.9), rgba(255,255,255,0));
  animation: rotateSoft 22s linear infinite;
}

.hero-main {
  min-width: 0;
  position: relative;
  z-index: 1;
}

.hero-main h1 {
  position: relative;
  margin: 0 0 6px;
  font-size: 30px;
  line-height: 1.18;
  letter-spacing: 0;
}

.ocean-badges {
  position: relative;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 16px;
}

.ocean-badges span,
.panel-hint span {
  display: inline-flex;
  align-items: center;
  min-height: 30px;
  padding: 6px 12px;
  border-radius: 6px;
  color: #0f5fcf;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(215, 231, 247, 0.9);
  font-size: 13px;
  box-shadow: 0 8px 18px rgba(22, 119, 255, 0.08);
}

.hero-nav {
  position: relative;
  z-index: 1;
  align-self: center;
}

.hero-nav button {
  min-width: 148px;
  min-height: 46px;
  border: 0 !important;
  border-radius: 999px !important;
  color: #0f5fcf !important;
  background: rgba(255, 255, 255, 0.95) !important;
  box-shadow: 0 14px 32px rgba(8, 64, 152, 0.22) !important;
}

.panel-hint {
  position: relative;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 14px;
}

.auth-section {
  position: relative;
  margin-bottom: 18px;
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
  box-shadow: 0 10px 24px rgba(34, 99, 174, 0.06);
}

.auth-section > .block,
.auth-section > div {
  overflow: visible !important;
}

.auth-section.user-section {
  min-height: 0;
  z-index: 40;
}

.auth-section.config-section {
  z-index: 20;
}

.auth-section-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 0 10px;
  color: #12243f;
  font-size: 18px;
  font-weight: 700;
}

.auth-section-title::before {
  content: "";
  width: 6px;
  height: 20px;
  border-radius: 999px;
  background: linear-gradient(180deg, #1677ff, #29a8ff);
  box-shadow: 0 8px 18px rgba(22, 119, 255, 0.22);
}

.auth-shell,
.chat-shell {
  position: relative;
  overflow: visible;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px;
  background: var(--panel);
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
}

.auth-shell .form,
.auth-shell .block,
.auth-shell .gap {
  overflow: visible !important;
}

.auth-shell {
  z-index: 30;
}

.chat-shell {
  z-index: 10;
}

.auth-shell::before,
.chat-shell::before {
  content: "";
  position: absolute;
  inset: 0 0 auto 0;
  height: 3px;
  pointer-events: none;
  border-radius: 8px 8px 0 0;
  background: linear-gradient(90deg, #1677ff, #29a8ff);
}

.auth-grid,
.action-row,
.tool-row {
  gap: 10px;
}

.auth-switch-row {
  margin-bottom: 12px;
}

.auth-switch-row button {
  min-height: 40px !important;
  font-weight: 700 !important;
}

.auth-grid {
  align-items: start;
  overflow: visible !important;
  row-gap: 8px !important;
  margin-top: 0 !important;
}

.login-grid {
  grid-template-columns: 1fr 1fr;
}

.auth-grid .wrap,
.auth-grid .container,
.auth-grid .form,
.auth-grid .block {
  margin-top: 0 !important;
}

.auth-grid > *,
.tool-row > *,
.action-row > * {
  min-width: 0;
}

.user-row {
  align-items: end;
}

.chatbot-panel {
  border: 1px solid var(--line) !important;
  border-radius: 8px !important;
  overflow: hidden;
  background: linear-gradient(180deg, #ffffff, #f7fbff) !important;
}

.chatbot-panel .message {
  border-radius: 8px !important;
  border: 1px solid rgba(22, 119, 255, 0.12) !important;
  box-shadow: 0 8px 20px rgba(31, 86, 144, 0.08);
}

.chatbot-panel .message.user {
  background: linear-gradient(135deg, #1677ff, #29a8ff) !important;
  color: #ffffff !important;
}

.chatbot-panel .message.bot,
.chatbot-panel .message.assistant {
  background: rgba(255, 255, 255, 0.98) !important;
  color: var(--ink) !important;
}

.message-input textarea {
  min-height: 78px !important;
}

.status-box textarea,
.detail-panel textarea {
  font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace !important;
}

.gradio-container button.primary,
.gradio-container button[variant="primary"] {
  border: 0 !important;
  color: #ffffff !important;
  background: linear-gradient(135deg, #1167ed, #1677ff 55%, #29a8ff) !important;
  box-shadow: 0 10px 24px rgba(22, 119, 255, 0.26) !important;
}

.gradio-container button {
  border-radius: 8px !important;
  border-color: var(--line) !important;
  color: var(--ink) !important;
  background: #ffffff !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.gradio-container button:hover {
  transform: translateY(-1px);
  border-color: rgba(22, 119, 255, 0.36) !important;
  box-shadow: 0 12px 24px rgba(22, 119, 255, 0.14) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .wrap,
.gradio-container .container {
  border-color: var(--line) !important;
  border-radius: 8px !important;
}

.gradio-container input:focus,
.gradio-container textarea:focus {
  border-color: rgba(22, 119, 255, 0.56) !important;
  box-shadow: 0 0 0 3px rgba(22, 119, 255, 0.1) !important;
}

.gradio-container label,
.gradio-container .label-wrap span {
  color: var(--muted) !important;
}

.auth-shell [role="listbox"],
.auth-shell .dropdown-options,
.auth-shell .options,
.auth-shell .options ul {
  z-index: 9999 !important;
}

.auth-shell,
.auth-section,
.auth-grid,
.auth-shell .wrap:has([role="listbox"]) {
  overflow: visible !important;
}

.auth-user-dropdown,
.auth-user-dropdown > *,
.auth-user-dropdown .wrap,
.auth-user-dropdown .container,
.auth-user-dropdown .form,
.auth-user-dropdown .block {
  margin: 0 !important;
}

.auth-shell [data-testid="textbox"] input[type="password"] {
  letter-spacing: 0.02em;
}

.field-message {
  min-height: 0;
}

.form-error,
.form-ok {
  margin-top: 8px;
  padding: 9px 11px;
  border-radius: 8px;
  font-size: 13px;
  line-height: 1.45;
}

.form-error {
  color: #b42318;
  background: #fff3f1;
  border: 1px solid #ffd1cc;
}

.form-ok {
  color: #116329;
  background: #eefbf3;
  border: 1px solid #bdeccb;
}

.register-link-button {
  margin-top: -6px;
}

.register-link-button button {
  width: auto !important;
  min-width: 0 !important;
  padding: 2px 4px !important;
  border: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
  color: #1677ff !important;
  font-size: 13px !important;
  font-weight: 600 !important;
}

.register-link-button button:hover {
  color: #0b5ed7 !important;
  text-decoration: underline;
}

.change-password-panel {
  margin-top: -4px;
  margin-bottom: 14px;
}

.report-panel {
  border-radius: 8px;
  overflow: hidden;
}

.resource-actions {
  margin-bottom: 14px;
  gap: 10px;
}

.resource-panel,
.session-summary-panel,
.session-report-panel {
  min-height: 0;
}

.inline-memory-panel {
  margin-bottom: 14px;
}

.fold-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
}

.fold-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: linear-gradient(180deg, #ffffff, #f7fbff);
  box-shadow: 0 12px 26px rgba(34, 99, 174, 0.08);
  overflow: hidden;
}

.fold-card summary {
  cursor: pointer;
  list-style: none;
  padding: 14px 16px;
  color: #10284c;
  font-size: 16px;
  font-weight: 760;
  border-bottom: 1px solid rgba(215, 231, 247, 0.78);
  background: rgba(255, 255, 255, 0.88);
}

.fold-card summary::-webkit-details-marker {
  display: none;
}

.fold-card summary::before {
  content: "+";
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  margin-right: 9px;
  border-radius: 999px;
  color: #ffffff;
  background: linear-gradient(135deg, #1677ff, #29a8ff);
  font-weight: 800;
}

.fold-card[open] summary::before {
  content: "-";
}

.fold-card-body {
  padding: 14px;
}

.fold-card .doc-card {
  margin-bottom: 0;
  box-shadow: none;
}

.consult-shell {
  margin-top: 0;
}

.consult-shell,
.consult-panel {
  background:
    linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246, 251, 255, 0.98));
}

.consult-shell {
  border-color: rgba(22, 119, 255, 0.18);
  box-shadow: 0 18px 42px rgba(32, 96, 170, 0.12), inset 0 1px 0 rgba(255,255,255,.7);
}

.consult-shell::before {
  background: linear-gradient(90deg, #1167ed, #29a8ff 60%, rgba(41,168,255,.55));
}

.hub-actions button {
  min-height: 40px;
  min-width: 132px;
  border-radius: 999px !important;
}

.history-session-select {
  margin-bottom: 12px;
}

.history-title-row {
  display: grid !important;
  grid-template-columns: minmax(0, 1fr) 168px;
  align-items: end !important;
  gap: 12px !important;
  margin: 0 0 14px !important;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
  box-shadow: 0 10px 24px rgba(34, 99, 174, 0.06);
}

.history-title-row > *,
.history-title-input,
.history-title-save {
  min-width: 0 !important;
}

.history-title-input {
  margin: 0 !important;
}

.history-title-input input,
.history-title-input textarea {
  min-height: 44px !important;
}

.history-title-save {
  align-self: end !important;
}

.history-title-save button {
  width: 100% !important;
  min-height: 44px !important;
  padding: 0 18px !important;
  font-size: 15px !important;
  font-weight: 700 !important;
}

.consult-panel .doc-card {
  background: linear-gradient(180deg, #ffffff, #f7fbff);
  border-color: rgba(215, 231, 247, 0.92);
}

.consult-panel .doc-title {
  color: #0f3e8a;
}

.consult-panel .doc-title::before {
  background: linear-gradient(180deg, #1677ff, #7bd8ff);
}

.consult-panel .pretty-list li > span,
.consult-panel .list-index {
  background: linear-gradient(135deg, #1677ff, #29a8ff);
}

.consult-panel .field-key {
  color: #1456c9;
}

.consult-panel .empty {
  background: #f8fcff;
}

.export-html-area {
  margin-top: 12px;
  padding: 14px 16px;
  border: 1px solid rgba(215, 231, 247, 0.9);
  border-radius: 8px;
  background: linear-gradient(180deg, #ffffff, #f6fbff);
}

.export-action-row {
  justify-content: flex-end;
  margin: 0;
}

.export-action-row button {
  min-width: 132px;
  border-radius: 999px !important;
}

.export-confirm-panel {
  margin-top: 10px;
}

.api-mode-radio {
  margin-top: 2px;
  margin-bottom: 8px;
}

.model-choice-row {
  display: block !important;
}

.model-choice-radio {
  width: 100%;
  margin-bottom: 10px;
}

.model-choice-radio .wrap,
.model-choice-radio .container,
.model-choice-radio .form {
  overflow: visible !important;
}

.model-choice-radio [role="radiogroup"] {
  display: grid !important;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px !important;
}

.model-choice-radio label {
  min-height: 44px;
}

.doc-card {
  margin: 0 0 14px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: linear-gradient(180deg, #ffffff, #f7fbff);
  box-shadow: 0 12px 26px rgba(34, 99, 174, 0.08);
}

.doc-title {
  display: flex;
  align-items: center;
  gap: 10px;
  color: #10284c;
  font-size: 18px;
  font-weight: 750;
  margin-bottom: 8px;
}

.doc-title::before {
  content: "";
  width: 8px;
  height: 22px;
  border-radius: 999px;
  background: linear-gradient(180deg, #1677ff, #29a8ff);
}

.doc-subtitle {
  color: var(--muted);
  margin-bottom: 12px;
  font-size: 13px;
}

.doc-body {
  display: grid;
  gap: 10px;
}

.json-row {
  display: grid;
  grid-template-columns: 160px minmax(0, 1fr);
  gap: 12px;
  padding: 12px;
  border: 1px solid rgba(215, 231, 247, 0.78);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.78);
}

.json-key {
  color: #1456c9;
  font-weight: 700;
  line-height: 1.6;
}

.json-value,
.json-text {
  color: #1b2d45;
  line-height: 1.7;
  word-break: break-word;
}

.json-list {
  margin: 0;
  padding-left: 0;
  list-style: none;
}

.json-list li {
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr);
  gap: 10px;
  margin: 8px 0;
  line-height: 1.7;
}

.list-index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 999px;
  color: #ffffff;
  background: #1677ff;
  font-size: 12px;
  font-weight: 700;
}

.list-content {
  min-width: 0;
}

.empty-card {
  padding: 14px;
  color: var(--muted);
  border: 1px dashed var(--line);
  border-radius: 8px;
  background: #fbfdff;
}

.doc-path {
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid var(--line);
  color: #7a8da6;
  font-size: 12px;
  word-break: break-all;
}

.two-report-cols {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.session-report-card .doc-card {
  box-shadow: none;
}

.full-report-shell {
  display: grid;
  gap: 14px;
}

@keyframes workspaceGrid {
  0% { background-position: 0 0, 0 0, 0 0; }
  100% { background-position: 32px 32px, 32px 32px, 0 0; }
}

@keyframes blueSheen {
  0%, 100% { background-position: 0% 0, 0 0; }
  50% { background-position: 100% 0, 0 0; }
}

@keyframes blueprintMove {
  0% { background-position: 0 0, 0 0; }
  100% { background-position: 22px 22px, -22px 22px; }
}

@keyframes rotateSoft {
  to { transform: rotate(360deg); }
}

@media (max-width: 720px) {
  .hero-layout {
    padding: 20px 18px;
  }

  .hero-main h1 {
    font-size: 24px;
  }

  .auth-shell,
  .chat-shell {
    padding: 12px;
  }

  .auth-section {
    padding: 14px;
  }

  .json-row {
    grid-template-columns: 1fr;
  }

  .two-report-cols {
    grid-template-columns: 1fr;
  }

  .history-title-row {
    grid-template-columns: 1fr;
  }

  .history-title-save button {
    min-height: 42px !important;
  }

  .fold-grid {
    grid-template-columns: 1fr;
  }

  .hero-layout {
    grid-template-columns: 1fr;
  }

  .hero-nav button {
    width: 100%;
  }
}
"""


def build_app() -> gr.Blocks:
    migrate_legacy_session_tree()
    ensure_seed_users()
    login_user_choices = collect_known_user_ids()
    with gr.Blocks(title="MemAgent Chat") as demo:
        state = gr.State(empty_state())
        browser_session = gr.BrowserState({}, storage_key="memagent_browser_session_v1")

        with gr.Row(elem_classes=["app-shell", "hero-layout"]):
            gr.HTML(
                """
                <div class="hero-main">
                  <h1>MemAgent 咨询系统</h1>
                  <div class="ocean-badges">
                    <span>连续咨询</span>
                    <span>长期记忆</span>
                    <span>阶段摘要</span>
                    <span>报告追踪</span>
                  </div>
                </div>
                """
            )
            with gr.Column(elem_classes=["hero-nav"]):
                consult_nav_btn = gr.Button("我的咨询")

        with gr.Column(visible=True, elem_classes=["auth-shell"]) as auth_shell:
            with gr.Row(elem_classes=["action-row", "auth-switch-row"]):
                login_mode_btn = gr.Button("登录", variant="primary")
                register_mode_btn = gr.Button("注册账户")
            with gr.Column(visible=True, elem_classes=["auth-page", "login-page"]) as auth_page:
                with gr.Group(elem_classes=["auth-section", "user-section"]):
                    gr.HTML('<div class="auth-section-title">登录</div>')
                    existing_user = gr.Dropdown(
                        choices=login_user_choices,
                        value=None,
                        label="已有用户",
                        allow_custom_value=False,
                        filterable=False,
                        visible=False,
                        elem_classes=["auth-user-dropdown"],
                    )
                    typed_user = gr.Textbox(
                        label="用户名称 / 用户 ID",
                        value="",
                        placeholder="请输入你的账号名称",
                        interactive=True,
                    )
                    with gr.Row(elem_classes=["auth-grid", "login-grid"]):
                        login_password = gr.Textbox(label="密码", type="password", interactive=True)
                    login_form_message = gr.HTML(elem_classes=["field-message"])
                    with gr.Row(elem_classes=["action-row", "login-action-row"]):
                        login_btn = gr.Button("登录", variant="primary")
                    with gr.Accordion("忘记密码 / 手机验证码登录", open=False, elem_classes=["change-password-panel"]):
                        phone_login_number = gr.Textbox(label="已绑定手机号", interactive=True)
                        with gr.Row(elem_classes=["action-row"]):
                            send_phone_login_code_btn = gr.Button("获取验证码")
                        phone_login_code = gr.Textbox(label="验证码", interactive=True)
                        with gr.Row(elem_classes=["auth-grid", "login-grid"]):
                            phone_new_password = gr.Textbox(label="新密码（可选）", type="password", interactive=True)
                            phone_confirm_password = gr.Textbox(label="确认新密码", type="password", interactive=True)
                        phone_login_message = gr.HTML(elem_classes=["field-message"])
                        with gr.Row(elem_classes=["action-row", "login-action-row"]):
                            phone_login_btn = gr.Button("验证码登录", variant="primary")
                login_agent_backend = gr.State("gpt")
                login_summary_backend = gr.State("gpt")
                login_supervisor_backend = gr.State("gpt")
                login_graph_backend = gr.State("gpt")
                login_router_backend = gr.State("gpt")
                auth_status = gr.Textbox(label="认证状态", interactive=False, lines=3, elem_classes=["status-box"])

            with gr.Column(visible=False, elem_classes=["auth-page", "register-page"]) as register_page:
                with gr.Group(elem_classes=["auth-section", "user-section"]):
                    gr.HTML('<div class="auth-section-title">注册账户</div>')
                    with gr.Row(elem_classes=["auth-grid", "login-grid"]):
                        register_user_id = gr.Textbox(label="用户名 / 用户 ID", interactive=True)
                        register_display_name = gr.Textbox(label="显示名称", interactive=True)
                    with gr.Row(elem_classes=["auth-grid", "login-grid"]):
                        register_password = gr.Textbox(label="密码", type="password", interactive=True)
                        register_confirm_password = gr.Textbox(label="确认密码", type="password", interactive=True)
                    with gr.Row(elem_classes=["action-row"]):
                        register_btn = gr.Button("注册账户", variant="primary")
                        back_to_login_btn = gr.Button("返回登录")
                register_status = gr.HTML(elem_classes=["field-message"])

        with gr.Column(visible=False, elem_classes=["chat-shell"]) as chat_page:
            with gr.Row(elem_classes=["user-row"]):
                current_user = gr.Textbox(label="当前用户", interactive=False)
            resource_panel = gr.HTML(label="资料展示", elem_classes=["resource-panel", "inline-memory-panel"])
            chatbot = gr.Chatbot(
                label="咨询对话",
                height=520,
                buttons=["copy", "copy_all"],
                layout="bubble",
                elem_classes=["chatbot-panel"],
            )
            user_message = gr.Textbox(
                label="输入消息",
                placeholder="写下此刻想聊的事...",
                lines=3,
                elem_classes=["message-input"],
            )
            with gr.Row(elem_classes=["action-row"]):
                send_btn = gr.Button("发送", variant="primary")
                end_btn = gr.Button("结束当前会话")
            chat_status = gr.Textbox(label="会话状态", interactive=False, lines=4, elem_classes=["status-box"])
            retrieval_panel = gr.HTML(label="本轮检索证据", elem_classes=["resource-panel", "retrieval-panel"])

        with gr.Column(visible=False, elem_classes=["chat-shell", "consult-shell"]) as consult_page:
            with gr.Row(elem_classes=["resource-actions", "hub-actions"]):
                hub_report_btn = gr.Button("完整报告")
                hub_history_btn = gr.Button("历史会话")
                chat_nav_btn = gr.Button("返回会话")
            history_session = gr.Dropdown(
                choices=[""],
                value="",
                label="会话编号",
                allow_custom_value=False,
                filterable=False,
                visible=False,
                elem_classes=["history-session-select"],
            )
            with gr.Row(visible=False, elem_classes=["history-title-row"]) as history_title_editor:
                history_session_title = gr.Textbox(
                    label="会话名称",
                    interactive=True,
                    elem_classes=["history-title-input"],
                )
                save_history_title_btn = gr.Button(
                    "保存名称",
                    variant="primary",
                    elem_classes=["history-title-save"],
                )
            consult_panel = gr.HTML(label="我的咨询", elem_classes=["resource-panel", "consult-panel"])
            with gr.Group(visible=False, elem_classes=["auth-section", "export-html-area"]) as export_html_area:
                with gr.Row(elem_classes=["action-row", "export-action-row"]):
                    export_html_btn = gr.Button("导出报告", variant="primary")
                with gr.Group(visible=False, elem_classes=["export-confirm-panel"]) as export_confirm_panel:
                    export_confirm_message = gr.HTML('<div class="form-ok">确认导出当前用户的完整 HTML 报告吗？</div>')
                    with gr.Row(elem_classes=["action-row"]):
                        confirm_export_html_btn = gr.Button("确认导出", variant="primary")
                        cancel_export_html_btn = gr.Button("取消")
                    export_html_file = gr.File(label="HTML 报告下载", visible=False)
            consult_detail_panel = gr.HTML(
                label="完整报告概览",
                visible=False,
                elem_classes=["resource-panel", "consult-panel", "consult-detail-panel"],
            )
            consult_status = gr.Textbox(label="咨询资料状态", interactive=False, lines=2, elem_classes=["status-box"])
            with gr.Group(visible=True, elem_classes=["auth-section", "config-section", "account-section"]):
                gr.HTML('<div class="auth-section-title">我的配置</div>')
                with gr.Accordion("模型配置", open=True, elem_classes=["change-password-panel"]):
                    with gr.Row(elem_classes=["auth-grid", "model-choice-row"]):
                        config_agent_backend = gr.Radio(
                            choices=MODEL_BACKEND_CHOICES,
                            value="gpt",
                            label="完整模型选择",
                            elem_classes=["model-choice-radio"],
                        )
                        config_summary_backend = gr.Dropdown(SUMMARY_BACKEND_CHOICES, value="gpt", label="Summary Agent", visible=False)
                    with gr.Row(elem_classes=["auth-grid", "login-grid"]):
                        config_graph_backend = gr.Dropdown(AGENT_BACKEND_CHOICES, value="gpt", label="Graph Extractor", visible=False)
                        config_router_backend = gr.Dropdown(AGENT_BACKEND_CHOICES, value="gpt", label="路由模型", visible=False)
                    config_api_mode = gr.Radio(
                        choices=API_MODE_CHOICES,
                        value="own",
                        label="API 使用方式",
                        elem_classes=["api-mode-radio"],
                    )
                    with gr.Row(elem_classes=["auth-grid", "login-grid"]):
                        config_api_provider = gr.Dropdown(API_PROVIDER_CHOICES, value=DEFAULT_API_PROVIDER, label="API 提供方", visible=False)
                    trial_quota_status = gr.HTML(_trial_quota_message({"trial_sessions_used": 0}), elem_classes=["field-message"])
                    with gr.Row(elem_classes=["auth-grid", "login-grid"]):
                        config_api_key = gr.Textbox(label="API Key", type="password", interactive=True)
                    with gr.Row(elem_classes=["auth-grid", "login-grid"]):
                        config_model_name = gr.Textbox(
                            label="API 模型名",
                            value=DEFAULT_API_MODEL_NAME,
                            interactive=False,
                        )
                    config_status = gr.HTML(elem_classes=["field-message"])
                    with gr.Row(elem_classes=["action-row"]):
                        save_config_btn = gr.Button("保存配置", variant="primary")
                    config_agent_backend.change(
                        fn=update_model_selection_preview,
                        inputs=[config_agent_backend, config_api_mode, state],
                        outputs=[config_api_key, config_model_name],
                        queue=False,
                    )
                    config_api_mode.change(
                        fn=update_model_selection_preview,
                        inputs=[config_agent_backend, config_api_mode, state],
                        outputs=[config_api_key, config_model_name],
                        queue=False,
                    )
            with gr.Group(elem_classes=["auth-section", "account-section"]):
                gr.HTML('<div class="auth-section-title">账号管理</div>')
                with gr.Accordion("修改密码", open=False, elem_classes=["change-password-panel"]):
                    with gr.Row(elem_classes=["auth-grid", "login-grid"]):
                        old_password = gr.Textbox(label="原密码", type="password", interactive=True)
                        new_password = gr.Textbox(label="新密码", type="password", interactive=True)
                        confirm_new_password = gr.Textbox(label="确认新密码", type="password", interactive=True)
                    change_password_message = gr.HTML(elem_classes=["field-message"])
                    with gr.Row(elem_classes=["action-row"]):
                        submit_change_password_btn = gr.Button("保存新密码", variant="primary")
                with gr.Accordion("手机号绑定", open=False, elem_classes=["change-password-panel"]):
                    bind_phone_input = gr.Textbox(label="手机号", interactive=True)
                    bind_phone_code = gr.Textbox(label="验证码", interactive=True)
                    bind_phone_message = gr.HTML(elem_classes=["field-message"])
                    with gr.Row(elem_classes=["action-row"]):
                        send_bind_phone_code_btn = gr.Button("获取验证码")
                        submit_bind_phone_btn = gr.Button("绑定手机号", variant="primary")
                with gr.Row(elem_classes=["action-row"]):
                    logout_btn = gr.Button("退出登录")

        background_timer = gr.Timer(4.0)

        background_timer.tick(
            fn=poll_background_finalize_state,
            inputs=[state],
            outputs=[state, chat_status, resource_panel, retrieval_panel],
            queue=False,
        )

        login_event = login_btn.click(
            fn=login_user,
            inputs=[existing_user, typed_user, login_password],
            outputs=[
                state,
                auth_status,
                auth_shell,
                auth_page,
                register_page,
                chat_page,
                current_user,
                resource_panel,
                history_session,
                consult_panel,
                consult_page,
                login_form_message,
                login_password,
                consult_nav_btn,
                login_supervisor_backend,
                config_graph_backend,
                login_router_backend,
                config_agent_backend,
                config_summary_backend,
                config_router_backend,
                config_api_provider,
                config_api_key,
                config_model_name,
                config_api_mode,
                trial_quota_status,
                bind_phone_input,
                bind_phone_message,
                retrieval_panel,
                chatbot,
            ],
            queue=False,
        )
        login_event.then(fn=remember_chat_page, inputs=[state], outputs=[browser_session], queue=False, show_progress="hidden")
        login_event.then(fn=activate_chat_view, inputs=[state], outputs=[chat_page, consult_page, chat_status], queue=False, show_progress="hidden")
        send_phone_login_code_btn.click(
            fn=request_phone_login_code,
            inputs=[phone_login_number],
            outputs=[phone_login_message],
            queue=False,
        )
        phone_login_event = phone_login_btn.click(
            fn=verify_phone_login,
            inputs=[phone_login_number, phone_login_code, phone_new_password, phone_confirm_password],
            outputs=[
                state,
                auth_status,
                auth_shell,
                auth_page,
                register_page,
                chat_page,
                current_user,
                resource_panel,
                history_session,
                consult_panel,
                consult_page,
                login_form_message,
                login_password,
                consult_nav_btn,
                login_supervisor_backend,
                config_graph_backend,
                login_router_backend,
                config_agent_backend,
                config_summary_backend,
                config_router_backend,
                config_api_provider,
                config_api_key,
                config_model_name,
                config_api_mode,
                trial_quota_status,
                bind_phone_input,
                bind_phone_message,
                retrieval_panel,
                chatbot,
            ],
            queue=False,
        )
        phone_login_event.then(fn=remember_chat_page, inputs=[state], outputs=[browser_session], queue=False, show_progress="hidden")
        phone_login_event.then(fn=activate_chat_view, inputs=[state], outputs=[chat_page, consult_page, chat_status], queue=False, show_progress="hidden")
        login_mode_btn.click(
            fn=open_login_page,
            inputs=[],
            outputs=[auth_shell, auth_page, register_page, auth_status],
            queue=False,
        )
        register_mode_btn.click(
            fn=open_register_page,
            inputs=[],
            outputs=[auth_shell, auth_page, register_page, register_status],
            queue=False,
        )
        register_btn.click(
            fn=register_and_refresh,
            inputs=[register_user_id, register_display_name, register_password, register_confirm_password],
            outputs=[
                register_status,
                existing_user,
                auth_shell,
                auth_page,
                register_page,
                register_user_id,
                register_display_name,
                register_password,
                register_confirm_password,
            ],
            queue=False,
        )
        back_to_login_btn.click(
            fn=open_login_page,
            inputs=[],
            outputs=[auth_shell, auth_page, register_page, auth_status],
            queue=False,
        )
        submit_change_password_btn.click(
            fn=change_password,
            inputs=[state, old_password, new_password, confirm_new_password],
            outputs=[change_password_message, old_password, new_password, confirm_new_password],
            queue=False,
        )
        send_bind_phone_code_btn.click(
            fn=request_bind_phone_code,
            inputs=[state, bind_phone_input],
            outputs=[bind_phone_message],
            queue=False,
        )
        submit_bind_phone_btn.click(
            fn=bind_phone_number,
            inputs=[state, bind_phone_input, bind_phone_code],
            outputs=[state, bind_phone_input, bind_phone_message],
            queue=False,
        )
        save_config_btn.click(
            fn=save_model_config,
            inputs=[
                state,
                config_agent_backend,
                config_api_mode,
                config_api_provider,
                config_api_key,
                config_model_name,
            ],
            outputs=[state, config_status, config_api_key, config_model_name, trial_quota_status],
            queue=False,
        )
        logout_event = logout_btn.click(
            fn=logout_user,
            inputs=[],
            outputs=[
                state,
                auth_status,
                current_user,
                user_message,
                auth_shell,
                auth_page,
                register_page,
                chat_page,
                consult_page,
                resource_panel,
                chat_status,
                consult_panel,
                history_session,
                consult_status,
                change_password_message,
                old_password,
                new_password,
                confirm_new_password,
                consult_nav_btn,
                retrieval_panel,
            ],
            queue=False,
        )
        logout_event.then(fn=clear_browser_session, inputs=[], outputs=[browser_session], queue=False, show_progress="hidden")

        send_btn.click(
            fn=send_message,
            inputs=[user_message, state],
            outputs=[user_message, state, chatbot, chat_status, retrieval_panel, browser_session],
            concurrency_limit=None,
        )
        user_message.submit(
            fn=send_message,
            inputs=[user_message, state],
            outputs=[user_message, state, chatbot, chat_status, retrieval_panel, browser_session],
            concurrency_limit=None,
        )
        end_btn.click(
            fn=end_session,
            inputs=[state],
            outputs=[state, chatbot, chat_status, resource_panel, retrieval_panel, browser_session],
            concurrency_limit=None,
        )
        chat_nav_event = chat_nav_btn.click(
            fn=open_chat_page,
            inputs=[state],
            outputs=[chat_page, consult_page, chat_status],
            queue=False,
            show_progress="hidden",
        )
        chat_nav_event.then(fn=remember_chat_page, inputs=[state], outputs=[browser_session], queue=False, show_progress="hidden")
        consult_nav_event = consult_nav_btn.click(
            fn=open_consultation_hub,
            inputs=[state],
            outputs=[
                chat_page,
                consult_page,
                consult_panel,
                consult_detail_panel,
                history_session,
                consult_status,
                export_html_area,
                export_confirm_panel,
                export_html_file,
                history_title_editor,
                save_history_title_btn,
            ],
            queue=False,
            show_progress="hidden",
        )
        consult_nav_event.then(fn=remember_consult_overview, inputs=[state], outputs=[browser_session], queue=False, show_progress="hidden")
        export_html_btn.click(
            fn=open_html_export_confirm,
            inputs=[state],
            outputs=[export_confirm_panel, export_confirm_message, export_html_file],
            queue=False,
        )
        cancel_export_html_btn.click(
            fn=cancel_html_export,
            inputs=[],
            outputs=[export_confirm_panel, export_confirm_message, export_html_file],
            queue=False,
        )
        confirm_export_html_btn.click(
            fn=confirm_html_export,
            inputs=[state],
            outputs=[export_confirm_panel, export_confirm_message, export_html_file],
            queue=False,
        )
        hub_report_event = hub_report_btn.click(
            fn=show_hub_full_report,
            inputs=[state],
            outputs=[
                consult_panel,
                history_session,
                consult_status,
                export_html_area,
                export_confirm_panel,
                export_html_file,
                consult_detail_panel,
                history_title_editor,
                save_history_title_btn,
            ],
            queue=False,
            show_progress="hidden",
        )
        hub_report_event.then(fn=remember_consult_report, inputs=[state], outputs=[browser_session], queue=False, show_progress="hidden")
        hub_history_event = hub_history_btn.click(
            fn=show_hub_history,
            inputs=[state],
            outputs=[
                state,
                consult_panel,
                history_session,
                consult_status,
                export_html_area,
                export_confirm_panel,
                export_html_file,
                consult_detail_panel,
                history_session_title,
                history_title_editor,
                save_history_title_btn,
            ],
            queue=False,
            show_progress="hidden",
        )
        hub_history_event.then(fn=remember_consult_history, inputs=[state, history_session], outputs=[browser_session], queue=False, show_progress="hidden")
        history_change_event = history_session.change(
            fn=select_history_session,
            inputs=[history_session, state],
            outputs=[
                state,
                consult_panel,
                consult_status,
                export_html_area,
                export_confirm_panel,
                export_html_file,
                consult_detail_panel,
                history_session_title,
                history_title_editor,
                save_history_title_btn,
            ],
            queue=False,
            show_progress="hidden",
        )
        history_change_event.then(fn=remember_consult_history, inputs=[state, history_session], outputs=[browser_session], queue=False, show_progress="hidden")
        save_history_title_event = save_history_title_btn.click(
            fn=save_history_session_title,
            inputs=[history_session, history_session_title, state],
            outputs=[
                state,
                history_session,
                consult_panel,
                consult_status,
                history_session_title,
            ],
            queue=False,
            show_progress="hidden",
        )
        save_history_title_event.then(fn=remember_consult_history, inputs=[state, history_session], outputs=[browser_session], queue=False, show_progress="hidden")
        demo.load(
            fn=restore_browser_session,
            inputs=[browser_session],
            outputs=[
                state,
                auth_status,
                auth_shell,
                auth_page,
                register_page,
                chat_page,
                current_user,
                resource_panel,
                history_session,
                consult_panel,
                consult_detail_panel,
                consult_page,
                login_form_message,
                login_password,
                consult_nav_btn,
                login_supervisor_backend,
                config_graph_backend,
                login_router_backend,
                config_agent_backend,
                config_summary_backend,
                config_router_backend,
                config_api_provider,
                config_api_key,
                config_model_name,
                config_api_mode,
                trial_quota_status,
                bind_phone_input,
                bind_phone_message,
                retrieval_panel,
                export_html_area,
                export_confirm_panel,
                export_html_file,
                chatbot,
                browser_session,
            ],
            queue=False,
        )
        demo.load(fn=warmup_ui_callback, inputs=[], outputs=[], queue=False)

    return demo


def find_available_port(host: str, preferred_port: int, attempts: int = 20) -> int:
    for port in range(preferred_port, preferred_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(f"Cannot find empty port in range: {preferred_port}-{preferred_port + attempts - 1}.")


def launch_app() -> None:
    demo = build_app()
    public_mode = os.getenv("MEMAGENT_PUBLIC", "0").strip().lower() in {"1", "true", "yes"}
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0" if public_mode else "127.0.0.1")
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "0").strip().lower() in {"1", "true", "yes"}
    inbrowser = os.getenv("GRADIO_INBROWSER", "0").strip().lower() in {"1", "true", "yes"}
    root_path = os.getenv("GRADIO_ROOT_PATH") or None
    ssl_keyfile = os.getenv("GRADIO_SSL_KEYFILE") or None
    ssl_certfile = os.getenv("GRADIO_SSL_CERTFILE") or None
    demo.launch(
        server_name=server_name,
        server_port=find_available_port(server_name, preferred_port),
        inbrowser=inbrowser,
        share=share,
        root_path=root_path,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        ssl_verify=False if ssl_certfile else True,
        show_error=True,
        css=APP_CSS,
    )
