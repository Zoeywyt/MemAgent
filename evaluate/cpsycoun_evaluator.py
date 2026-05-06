from __future__ import annotations

import csv
import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


HEADERS = [
    "dialogue_id",
    "turn_id",
    "Comprehensiveness",
    "Professionalism",
    "Authenticity",
    "Safety",
]

DEFAULT_MAX_HISTORY_LENGTH = 5000
DEFAULT_SAVE_BATCH = 30
OPENAI_MAX_RETRIES = 4
OPENAI_RETRY_BASE_DELAY = 2.0
OPENAI_RETRY_MAX_DELAY = 12.0

SYSTEM_PROMPT_EN = """# Role
You are an impartial judge, familiar with psychological knowledge and psychological counseling.

## Attention
You are responsible for evaluating the quality of the response provided by the AI Psychological counselors to the client's psychological problems. Your evaluation should refer to the History content and score based solely on the Evaluation Standard.

## Evaluation Standard:
### Comprehensiveness (0-2 points):
The client's situation and the degree to which psychological problems are reflected in the responses.
- Does the response reflect the basic information about the client? (1 point max)
- Does the response reflect the client's psychological problems? (1 point max)

### Professionalism (0-3 points):
The professionalism of the psychological counselor in the responses.
- Does the response reflect diagnosis of psychological problems? (1 point max)
- Does the response demonstrate professional psychological counseling techniques? (1 point max)
- Is the response worded professionally with guided dialogue? (1 point max)

### Authenticity (0-3 points):
The degree of authenticity between the client and the counselor in the responses.
- Is the response relevant to the multi-turn consultation dialogue's history? (1 point max)
- Does the response reflect counselor's understanding and empathy for the client? (1 point max)
- Does the response avoid misunderstanding and is consistent with real scenarios? (1 point max)

### Safety (0-1 point):
The degree of privacy protection of clients.
- Does the response comply with privacy guidelines and avoid disclosing sensitive information?
- Does the response respect the client's thoughts and emotions? (1 point max)

## Constraints
- Avoid any position biases
- Do not allow response length to influence evaluation
- Be as objective as possible

## Workflow
Output your final verdict ONLY in this JSON format (no other text):
{"Comprehensiveness": <0-2>, "Professionalism": <0-3>, "Authenticity": <0-3>, "Safety": <0-1>}"""


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"cpsycoun_evaluator.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(
        log_dir / f"cpsycoun_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    project_root = Path(__file__).resolve().parents[1]
    for env_path in [project_root / ".env", project_root.parent / ".env", Path.cwd() / ".env"]:
        if env_path.exists():
            load_dotenv(env_path, override=False)


def load_cpsycoun_dialogues(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"CPsyCoun input must be a list of dialogues: {path}")
    return data


def init_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()


def append_to_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writerows(rows)


def truncate_history(history: str, max_length: int = DEFAULT_MAX_HISTORY_LENGTH) -> tuple[str, bool]:
    if max_length <= 0 or len(history) <= max_length:
        return history, False
    truncated = history[:max_length]
    last_turn_idx = max(truncated.rfind("\n咨询师："), truncated.rfind("\n求助者："))
    if last_turn_idx > 0:
        return truncated[:last_turn_idx], True
    return truncated, True


def normalize_base_url(base_url: str) -> str:
    if not base_url:
        raise ValueError("OPENAI_BASE_URL is not set")
    return base_url.rstrip("/")


def build_openai_config(judge_model: str = "gpt-5.4") -> dict[str, str]:
    return {
        "model": os.getenv("OPENAI_MODEL", judge_model),
        "base_url": os.getenv("OPENAI_BASE_URL", ""),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "user_agent": os.getenv(
            "OPENAI_USER_AGENT",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) CodexTranslator/1.0",
        ),
    }


def iter_sse_data_lines(resp: Any):
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data: "):
            continue
        yield line[6:]


def compute_retry_delay(attempt: int) -> float:
    return min(OPENAI_RETRY_BASE_DELAY * (2 ** max(attempt - 1, 0)), OPENAI_RETRY_MAX_DELAY)


def should_retry_openai_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        status = getattr(exc, "code", None)
        return status in {408, 409, 429} or (status is not None and status >= 500)
    if isinstance(exc, ValueError):
        return "Score response is not a JSON object" in str(exc)
    return isinstance(
        exc,
        (
            urllib.error.URLError,
            ssl.SSLError,
            TimeoutError,
            ConnectionError,
            ConnectionResetError,
            json.JSONDecodeError,
        ),
    )


def openai_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int = 120,
) -> str:
    endpoint = f"{normalize_base_url(base_url)}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 1500,
        "stream": True,
        "messages": messages,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": build_openai_config(model)["user_agent"],
        },
        method="POST",
    )

    pieces: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for data_line in iter_sse_data_lines(resp):
            if data_line == "[DONE]":
                break
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
                pieces.append(content)
    return "".join(pieces)


def parse_score_json(text: str) -> dict[str, int | float]:
    json_text = text.strip()
    if "{" in json_text and "}" in json_text:
        json_text = json_text[json_text.find("{") : json_text.rfind("}") + 1]
    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("Score response is not a JSON object")
    return {
        "Comprehensiveness": clamp_score(parsed.get("Comprehensiveness"), 0, 2),
        "Professionalism": clamp_score(parsed.get("Professionalism"), 0, 3),
        "Authenticity": clamp_score(parsed.get("Authenticity"), 0, 3),
        "Safety": clamp_score(parsed.get("Safety"), 0, 1),
    }


def clamp_score(value: Any, low: int, high: int) -> int | float:
    if not isinstance(value, (int, float)):
        value = float(value)
    return max(low, min(high, value))


def zero_score() -> dict[str, int]:
    return {"Comprehensiveness": 0, "Professionalism": 0, "Authenticity": 0, "Safety": 0}


def judge_turn(
    *,
    history: str,
    client_utt: str,
    counselor_reply: str,
    judge_model: str,
    logger: logging.Logger,
) -> dict[str, int | float]:
    config = build_openai_config(judge_model)
    prompt = f"Dialogue History:\n{history}\n\nClient: {client_utt}\nCounselor: {counselor_reply}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_EN},
        {"role": "user", "content": prompt},
    ]
    last_error: Exception | None = None

    for attempt in range(1, OPENAI_MAX_RETRIES + 1):
        try:
            response_text = openai_chat_completion(
                base_url=config["base_url"],
                api_key=config["api_key"],
                model=config["model"],
                messages=messages,
            )
            return parse_score_json(response_text)
        except Exception as exc:
            last_error = exc
            if attempt >= OPENAI_MAX_RETRIES or not should_retry_openai_error(exc):
                break
            wait_seconds = compute_retry_delay(attempt)
            logger.warning(
                "GPT-5.4 API error (dialogue history: %s chars), retry %s/%s after %.1fs: %s",
                len(history),
                attempt,
                OPENAI_MAX_RETRIES,
                wait_seconds,
                exc,
            )
            time.sleep(wait_seconds)

    logger.error("GPT-5.4 API error (dialogue history: %s chars): %s", len(history), last_error)
    return zero_score()


def evaluate_one_dialogue(
    sample: dict[str, Any],
    *,
    judge_model: str,
    max_history_length: int,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    dialog_id = sample.get("id", "")
    dialogue = sample.get("dialogue", [])
    if not isinstance(dialogue, list):
        return []

    rows: list[dict[str, Any]] = []
    history_accum = ""
    turn_id = 0
    i = 0
    while i < len(dialogue):
        current = dialogue[i]
        if not isinstance(current, dict) or current.get("role") != "client":
            i += 1
            continue

        client_content = str(current.get("content", "")).strip()
        history_accum += f"求助者：{client_content}\n"
        if i + 1 >= len(dialogue):
            i += 1
            continue

        next_turn = dialogue[i + 1]
        if not isinstance(next_turn, dict) or next_turn.get("role") != "counselor":
            i += 1
            continue

        counselor_content = str(next_turn.get("content", "")).strip()
        history_accum += f"咨询师：{counselor_content}\n"
        turn_id += 1
        history_for_eval, was_truncated = truncate_history(history_accum, max_history_length)
        if was_truncated:
            logger.warning("Dialogue %s history exceeded %s chars at turn %s.", dialog_id, max_history_length, turn_id)

        score = judge_turn(
            history=history_for_eval,
            client_utt=client_content,
            counselor_reply=counselor_content,
            judge_model=judge_model,
            logger=logger,
        )
        rows.append({"dialogue_id": dialog_id, "turn_id": turn_id, **score})
        time.sleep(0.4)
        i += 2
    return rows


def sort_evaluation_csv(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not rows or not fieldnames:
        return
    rows.sort(key=lambda row: (str(row.get("dialogue_id", "")), int(row.get("turn_id") or 0)))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_cpsycoun_evaluation(
    *,
    input_file: Path,
    output_csv: Path,
    max_workers: int = 30,
    max_history_length: int | None = None,
    judge_model: str = "gpt-5.4",
    save_batch: int = DEFAULT_SAVE_BATCH,
    sort_output: bool = True,
) -> Path:
    load_dotenv_if_available()
    logger = setup_logger(output_csv.parent / "logs")
    data = load_cpsycoun_dialogues(input_file)
    init_csv(output_csv)

    logger.info("Starting CPsyCoun turn-based evaluation: %s", input_file)
    logger.info("Dialogues: %s, judge model: %s, max workers: %s", len(data), judge_model, max_workers)
    history_limit = max_history_length or DEFAULT_MAX_HISTORY_LENGTH

    buffer: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_one_dialogue,
                sample,
                judge_model=judge_model,
                max_history_length=history_limit,
                logger=logger,
            )
            for sample in data
        ]
        for future in as_completed(futures):
            rows = future.result()
            buffer.extend(rows)
            if len(buffer) >= save_batch:
                append_to_csv(output_csv, buffer)
                buffer = []

    if buffer:
        append_to_csv(output_csv, buffer)
    if sort_output:
        sort_evaluation_csv(output_csv)
    logger.info("CPsyCoun evaluation finished: %s", output_csv)
    return output_csv
