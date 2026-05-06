from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "test_outputs"
CONSULTING_REPORTS_DIR = OUTPUT_ROOT / "consulting_reports"
SESSIONS_SUMMARIES_DIR = OUTPUT_ROOT / "sessions_summaries"
TRENDS_DIR = OUTPUT_ROOT / "trends"
WHOLE_SUMMARIES_DIR = OUTPUT_ROOT / "whole_summaries"


REPORT_FIELD_LABELS = {
    "emotion_trend": "情绪变化轨迹",
    "key_progress": "关键进展",
    "risk_note": "风险评估",
    "treatment_phase": "咨询阶段",
    "next_focus": "下一阶段重点",
}


def h(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def safe_user_id(user_id: str) -> str:
    text = re.sub(r"\s+", "_", str(user_id or "").strip())
    text = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown_user"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def pick_latest(paths: Iterable[Path]) -> Optional[Path]:
    existing = [path for path in paths if path.exists()]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None


def discover_results_file(explicit: Optional[str] = None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"找不到结果文件: {path}")
        return path
    latest = pick_latest(CONSULTING_REPORTS_DIR.glob("*_results.json"))
    if latest is None:
        latest = pick_latest(CONSULTING_REPORTS_DIR.glob("*_report.json"))
    if latest is None:
        raise FileNotFoundError("未找到 consulting_reports 下的报告 JSON 文件")
    return latest


def sessions_summaries_path(user_id: str) -> Path:
    return SESSIONS_SUMMARIES_DIR / f"{safe_user_id(user_id)}.json"


def trends_path(user_id: str) -> Path:
    return TRENDS_DIR / f"{safe_user_id(user_id)}.json"


def whole_summary_path(user_id: str) -> Path:
    return WHOLE_SUMMARIES_DIR / f"{safe_user_id(user_id)}_summary.json"


def normalize_topic(topic: Any) -> str:
    if isinstance(topic, list):
        return "、".join(str(item) for item in topic if str(item).strip())
    return str(topic or "")


def normalize_summary(summary: Any) -> Dict[str, Any]:
    if isinstance(summary, dict):
        return summary
    text = str(summary or "").strip()
    if not text:
        return {}
    fields: Dict[str, Any] = {}
    for label in ["主题", "背景", "总结", "会话总结", "风险评估"]:
        pattern = rf"{label}\s*[：:]\s*"
        match = re.search(pattern, text)
        if not match:
            continue
        start = match.end()
        next_match = None
        for next_label in ["主题", "背景", "总结", "会话总结", "风险评估"]:
            if next_label == label:
                continue
            candidate = re.search(rf"\n\s*{next_label}\s*[：:]\s*", text[start:])
            if candidate and (next_match is None or candidate.start() < next_match.start()):
                next_match = candidate
        fields[label] = text[start : start + next_match.start()].strip() if next_match else text[start:].strip()
    return fields or {"摘要": text}


def localize_report(report: Any) -> Dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    return {REPORT_FIELD_LABELS.get(str(key), str(key)): value for key, value in report.items()}


def render_value(value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return "<div class='empty'>暂无内容</div>"
        return "".join(
            f"<div class='field'><div class='field-key'>{h(key)}</div><div class='field-value'>{render_value(item)}</div></div>"
            for key, item in value.items()
        )
    if isinstance(value, list):
        if not value:
            return "<div class='empty'>暂无内容</div>"
        return "<ol class='pretty-list'>" + "".join(
            f"<li><span>{index}</span><div>{render_value(item)}</div></li>"
            for index, item in enumerate(value, start=1)
        ) + "</ol>"
    return f"<div class='text'>{h(value).replace(chr(10), '<br>') or '暂无内容'}</div>"


def card(title: str, body: Any, subtitle: str = "") -> str:
    sub = f"<div class='card-subtitle'>{h(subtitle)}</div>" if subtitle else ""
    return f"<section class='card'><h2>{h(title)}</h2>{sub}<div class='card-body'>{render_value(body)}</div></section>"


def metric(label: str, value: Any) -> str:
    return f"<div class='metric'><div class='metric-label'>{h(label)}</div><div class='metric-value'>{h(value)}</div></div>"


def load_split_outputs(user_id: str, report_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    l1 = read_json(whole_summary_path(user_id), None)
    if not isinstance(l1, dict):
        l1 = report_data.get("final_l1_summary") or {}

    summary_doc = read_json(sessions_summaries_path(user_id), {})
    summaries = summary_doc.get("sessions", []) if isinstance(summary_doc, dict) else []
    if not summaries:
        summaries = report_data.get("all_l2_summaries", []) or []

    trends_doc = read_json(trends_path(user_id), {})
    reports = trends_doc.get("reports", []) if isinstance(trends_doc, dict) else []
    if not reports and report_data.get("latest_treatment_report"):
        reports = [{"session_id": "最新报告", "report": report_data.get("latest_treatment_report")}]
    return l1, summaries, reports


def count_l3(report_data: Dict[str, Any]) -> Tuple[int, int]:
    total = 0
    stored = 0
    for session in report_data.get("session_results", []) or []:
        for record in session.get("l3_records", []) or []:
            total += 1
            stored += int((record.get("l3_memory_result") or {}).get("stored", 0) or 0)
    return total, stored


def build_report_cards(summaries: List[Dict[str, Any]], reports: List[Dict[str, Any]]) -> str:
    report_by_session = {
        str(item.get("session_id", "")): item.get("report", {})
        for item in reports
        if isinstance(item, dict)
    }
    cards: List[str] = []
    for index, item in enumerate(summaries, start=1):
        if not isinstance(item, dict):
            continue
        session_id = str(item.get("session_id") or f"Session{index}")
        title = normalize_topic(item.get("topic")) or session_id
        summary_body = normalize_summary(item.get("summary"))
        report_body = localize_report(report_by_session.get(session_id, {}))
        cards.append(
            "<section class='session-block'>"
            f"<div class='session-head'><div><div class='session-id'>{h(session_id)}</div><h3>{h(title)}</h3></div>"
            f"<span>{h(item.get('archived_at') or item.get('generated_at') or '')}</span></div>"
            "<div class='split'>"
            f"{card('L2 历史会话总结', summary_body)}"
            f"{card('本次督导师报告', report_body)}"
            "</div>"
            "</section>"
        )
    return "".join(cards) if cards else card("L2 历史会话总结 + 督导师报告", {})


def build_html(data: Dict[str, Any], results_path: Path, checkpoint_path: Optional[Path] = None, summary_path: Optional[Path] = None) -> str:
    user_id = str(data.get("user_id") or results_path.stem.split("_")[0])
    l1, summaries, reports = load_split_outputs(user_id, data)
    l3_turns, l3_stored = count_l3(data)
    session_count = len(summaries) or len(data.get("session_results", []) or [])
    graph_relations = sum(
        len(((session.get("end_result") or {}).get("graph_data") or {}).get("relations", []) or [])
        for session in data.get("session_results", []) or []
    )
    metrics = "".join(
        [
            metric("用户", user_id),
            metric("Session 数", session_count),
            metric("L1 画像", 1 if l1 else 0),
            metric("L2 历史摘要", len(summaries)),
            metric("督导师报告", len(reports)),
            metric("L3 处理轮次", l3_turns),
            metric("L3 实际写入", l3_stored),
            metric("Graph 关系", graph_relations),
        ]
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{h(user_id)} 咨询记忆报告</title>
  <style>
    :root {{
      --blue: #1677ff;
      --cyan: #29a8ff;
      --ink: #13233a;
      --muted: #607289;
      --line: #d7e7f7;
      --bg: #f6fbff;
      --panel: #ffffff;
      --shadow: 0 18px 42px rgba(32, 96, 170, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        linear-gradient(90deg, rgba(22,119,255,.035) 1px, transparent 1px),
        linear-gradient(0deg, rgba(22,119,255,.035) 1px, transparent 1px),
        linear-gradient(180deg, #fbfdff 0%, var(--bg) 55%, #fff 100%);
      background-size: 32px 32px, 32px 32px, 100% 100%;
      line-height: 1.72;
    }}
    .wrap {{ width: min(1180px, calc(100vw - 36px)); margin: 28px auto 56px; }}
    .hero {{
      color: #fff;
      padding: 26px 30px;
      border-radius: 8px;
      background: linear-gradient(135deg, #1167ed 0%, #1677ff 48%, #22b7ff 100%);
      box-shadow: 0 18px 48px rgba(22,119,255,.2);
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 30px; line-height: 1.2; }}
    .hero p {{ margin: 0; color: rgba(255,255,255,.86); }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 16px 0; }}
    .metric, .card, .session-block {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,.96);
      box-shadow: var(--shadow);
    }}
    .metric {{ padding: 15px 16px; }}
    .metric-label {{ color: var(--muted); font-size: 13px; margin-bottom: 6px; }}
    .metric-value {{ color: #10284c; font-size: 24px; font-weight: 760; }}
    .card {{ padding: 18px; margin-bottom: 14px; box-shadow: 0 12px 28px rgba(32,96,170,.08); }}
    .card h2 {{ display: flex; align-items: center; gap: 10px; margin: 0 0 10px; font-size: 20px; }}
    .card h2::before {{ content: ""; width: 7px; height: 22px; border-radius: 999px; background: linear-gradient(180deg, var(--blue), var(--cyan)); }}
    .card-subtitle {{ color: var(--muted); margin-bottom: 12px; font-size: 13px; }}
    .field {{ display: grid; grid-template-columns: 150px minmax(0,1fr); gap: 12px; padding: 12px; border: 1px solid rgba(215,231,247,.78); border-radius: 8px; margin-bottom: 10px; background: rgba(255,255,255,.78); }}
    .field-key {{ color: #1456c9; font-weight: 720; }}
    .text {{ white-space: pre-wrap; word-break: break-word; }}
    .pretty-list {{ margin: 0; padding: 0; list-style: none; }}
    .pretty-list li {{ display: grid; grid-template-columns: 28px minmax(0,1fr); gap: 10px; margin: 8px 0; }}
    .pretty-list li > span {{ display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; border-radius: 999px; color: #fff; background: var(--blue); font-size: 12px; font-weight: 800; }}
    .empty {{ padding: 12px; color: var(--muted); border: 1px dashed var(--line); border-radius: 8px; }}
    .session-block {{ margin: 16px 0; padding: 18px; }}
    .session-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 14px; }}
    .session-id {{ color: var(--blue); font-weight: 760; }}
    .session-head h3 {{ margin: 2px 0 0; font-size: 18px; }}
    .session-head span {{ color: var(--muted); font-size: 12px; white-space: nowrap; }}
    .split {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .split .card {{ margin: 0; box-shadow: none; }}
    @media (max-width: 820px) {{ .split, .field {{ grid-template-columns: 1fr; }} .session-head {{ display: block; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>咨询记忆报告</h1>
      <p>当前报告保留各阶段记忆数、L1 画像、L2 历史会话总结，以及每一次会话对应的督导师报告。</p>
    </section>
    <section class="metrics">{metrics}</section>
    {card("L1 长期画像", l1)}
    {build_report_cards(summaries, reports)}
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="把咨询记忆 JSON 渲染为当前统一 HTML 报告")
    parser.add_argument("--input", help="指定 consulting report JSON 路径")
    parser.add_argument("--output", help="输出 HTML 路径")
    args = parser.parse_args()

    results_path = discover_results_file(args.input)
    data = read_json(results_path, {})
    user_id = str(data.get("user_id") or results_path.stem.split("_")[0])
    summary_path = whole_summary_path(user_id)
    output_path = Path(args.output) if args.output else results_path.with_name(results_path.stem + "_demo.html")
    html_text = build_html(data, results_path, None, summary_path)
    write_text(output_path, html_text)
    print(f"已生成可视化页面: {output_path.resolve()}")


if __name__ == "__main__":
    main()
