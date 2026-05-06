from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def to_clickable_uri(path: Path) -> str:
    return path.resolve().as_uri()


def pick_latest(paths: Iterable[Path]) -> Optional[Path]:
    items = sorted((p for p in paths if p.exists()), key=lambda p: p.stat().st_mtime, reverse=True)
    return items[0] if items else None


def discover_results_file(explicit: Optional[str] = None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"找不到结果文件: {path}")
        return path

    candidates = list(Path(".").rglob("*replay_results.json"))
    latest = pick_latest(candidates)
    if latest is None:
        raise FileNotFoundError("未找到 replay_results.json，请先运行 replay 脚本")
    return latest


def discover_checkpoint_file(results_path: Path) -> Optional[Path]:
    guess = results_path.with_name(results_path.name.replace("_replay_results.json", "_replay_checkpoint.json"))
    if guess.exists():
        return guess
    candidates = list(results_path.parent.glob("*replay_checkpoint.json"))
    return pick_latest(candidates)


def discover_summary_file(user_id: str) -> Optional[Path]:
    exact = Path("multi_agent/multi_agent_user_summaries") / f"{user_id}_summary.json"
    if exact.exists():
        return exact

    alternatives = list(Path(".").rglob("*_summary.json"))
    for path in alternatives:
        if path.name.startswith(user_id):
            return path
    return pick_latest(alternatives)


def h(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def get_first(mapping: Optional[Dict[str, Any]], keys: List[str], default: str = "") -> str:
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return str(mapping[key])
    for value in mapping.values():
        if value not in (None, ""):
            return str(value)
    return default


def badge(text: str, tone: str = "neutral") -> str:
    return f'<span class="badge badge-{tone}">{h(text)}</span>'


def render_list(items: Iterable[str], empty: str = "无") -> str:
    rows = [f"<li>{h(item)}</li>" for item in items if str(item).strip()]
    if not rows:
        return f"<div class='muted'>{h(empty)}</div>"
    return "<ul>" + "".join(rows) + "</ul>"


def render_kv_table(rows: List[Tuple[str, Any]]) -> str:
    body = "".join(f"<tr><th>{h(label)}</th><td>{h(value)}</td></tr>" for label, value in rows)
    return f"<table class='kv-table'>{body}</table>"


def render_relation_table(relations: List[Dict[str, Any]], limit: int = 12) -> str:
    rows = "".join(
        f"<tr><td>{h(item.get('source', ''))}</td><td>{h(item.get('relation', item.get('relationship', '')))}</td><td>{h(item.get('destination', item.get('target', '')))}</td></tr>"
        for item in relations[:limit]
    )
    if not rows:
        rows = "<tr><td colspan='3' class='muted'>无图关系</td></tr>"
    return f"""
    <table class="grid-table">
      <tr><th>Source</th><th>Relation</th><th>Destination</th></tr>
      {rows}
    </table>
    """


def build_overview_cards(data: Dict[str, Any]) -> str:
    session_results = data.get("session_results", [])
    all_l2 = data.get("all_l2_summaries", [])
    total_turns = sum(len(session.get("l3_records", [])) for session in session_results)
    total_l3_stored = sum(
        int(record.get("l3_memory_result", {}).get("stored", 0))
        for session in session_results
        for record in session.get("l3_records", [])
    )
    total_relations = sum(
        len(session.get("end_result", {}).get("graph_data", {}).get("relations", []))
        for session in session_results
    )

    cards = [
        ("用户", data.get("user_id", "未知")),
        ("Session 数", len(session_results)),
        ("L2 存储数", len(all_l2)),
        ("L3 轮次", total_turns),
        ("L3 实际写入", total_l3_stored),
        ("Graph 关系数", total_relations),
        ("状态", data.get("status", "unknown")),
    ]
    return "".join(
        f"<div class='metric-card'><div class='metric-label'>{h(label)}</div><div class='metric-value'>{h(value)}</div></div>"
        for label, value in cards
    )


def render_architecture_panel(data: Dict[str, Any]) -> str:
    l2_count = len(data.get("all_l2_summaries", []))
    l3_count = sum(len(session.get("l3_records", [])) for session in data.get("session_results", []))
    session_count = len(data.get("session_results", []))
    return f"""
    <section class="panel">
      <h2>记忆框架总览</h2>
      <div class="arch-grid">
        <div class="arch-box arch-input">
          <div class="arch-title">原始对话</div>
          <div class="arch-text">共 {session_count} 个 Session，逐轮输入用户与咨询师对话。</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-box arch-l3">
          <div class="arch-title">L3 片段记忆</div>
          <div class="arch-text">每轮判断是否值得存储，当前共处理 {l3_count} 轮。</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-box arch-l2">
          <div class="arch-title">L2 会话摘要</div>
          <div class="arch-text">每个 Session 结束后生成摘要，当前共 {l2_count} 条。</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-box arch-l1">
          <div class="arch-title">L1 长期画像</div>
          <div class="arch-text">跨 Session 累积更新，沉淀长期主题、背景与总体趋势。</div>
        </div>
      </div>
      <div class="arch-grid arch-grid-bottom">
        <div class="arch-box arch-graph">
          <div class="arch-title">session_graph</div>
          <div class="arch-text">这里按你的新展示口径，只强调关系入库与关系检索，不展示实体入库。</div>
        </div>
        <div class="arch-box arch-report">
          <div class="arch-title">督导师报告</div>
          <div class="arch-text">基于 L1 与全部 L2 生成阶段性评估，用于解释咨询进展。</div>
        </div>
        <div class="arch-box arch-retrieval">
          <div class="arch-title">后台检索</div>
          <div class="arch-text">展示口径调整为仅检索 L3 与 graph_relations，不展示 L2 与 session_graph 实体召回。</div>
        </div>
      </div>
      <div class="explain-box">
        <div class="explain-title"></div>
        <ol>
          <li>L3 记忆模块：每个 turn 先经过 L3 记忆入库的逻辑判断，只有“值得长期保留”的片段才会写入。</li>
          <li>session_graph模块：每个 turn 同时做图谱抽取，把用户的症状、情绪、兴趣、冲突和目标等按“实体-关系”组织。</li>
          <li>L2 记忆模块：每个 Session 结束后，本次会话会压缩成 L2 摘要。</li>
          <li>L1 记忆模块：用户画像，基于每次L2摘要 ，同步更新 L1 长期画像。</li>
          <li>督导师模块基于 L1 和全部 L2 纵向分析治疗进展。</li>
          <li>检索阶段同时搜索按向量相似度分数排序的文本记忆：会话片段、会话摘及按 BM25 排序的Kuzu图推理关系，保证文本的细节化及图关系的全局性。</li>
        </ol>
      </div>
    </section>
    """


def render_l1_panel(data: Dict[str, Any], summary_path: Optional[Path]) -> str:
    l1 = data.get("final_l1_summary", {}) or {}
    rows = [
        ("主题 / 长期主线", get_first(l1, ["主题", "topic"])),
        ("背景 / 长期背景", get_first(l1, ["背景", "background"])),
        ("会话总结 / 长期综合总结", get_first(l1, ["会话总结", "summary"])),
    ]
    extra = f"<div class='file-note'>来源文件: {h(summary_path) if summary_path else '未找到 summary 文件'}</div>"
    return f"""
    <section class="panel">
      <h2>L1 长期记忆</h2>
      {extra}
      {render_kv_table(rows)}
    </section>
    """


def render_l2_panel(data: Dict[str, Any]) -> str:
    cards: List[str] = []
    for item in data.get("all_l2_summaries", []):
        cards.append(
            f"""
            <div class="session-card">
              <div class="session-card-head">
                <div class="session-name">{h(item.get('session_id', 'unknown'))}</div>
                {badge('L2 存储', 'l2')}
              </div>
              <div class="session-topic">{h(item.get('topic', ''))}</div>
              <div class="session-summary">{h(item.get('summary', ''))}</div>
            </div>
            """
        )
    if not cards:
        cards.append("<div class='muted'>未找到 L2 摘要。</div>")
    return f"""
    <section class="panel">
      <h2>L2 会话级摘要</h2>
      <div class="card-grid">{''.join(cards)}</div>
    </section>
    """


def render_l3_records(records: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for record in records:
        result = record.get("l3_memory_result", {}) or {}
        facts = result.get("facts", []) or []
        actions = result.get("actions", []) or []
        action_rows: List[str] = []
        for action in actions:
            event = str(action.get("event", "NONE")).upper()
            tone = {"ADD": "green", "UPDATE": "blue", "DELETE": "red", "NONE": "gray"}.get(event, "gray")
            action_rows.append(
                "<tr>"
                f"<td>{badge(event, tone)}</td>"
                f"<td>{h(action.get('text', ''))}</td>"
                f"<td>{h(action.get('old_memory', ''))}</td>"
                "</tr>"
            )
        if not action_rows:
            action_rows.append("<tr><td colspan='3' class='muted'>本轮未触发写入动作</td></tr>")

        blocks.append(
            f"""
            <details class="turn-box">
              <summary>
                <span>Turn {h(record.get('turn_index', ''))}</span>
                <span>{badge(f"stored={result.get('stored', 0)}", 'l3')}</span>
              </summary>
              <div class="turn-inner">
                <div class="turn-role"><strong>User:</strong> {h(record.get('user', ''))}</div>
                <div class="turn-role"><strong>Assistant:</strong> {h(record.get('assistant', ''))}</div>
                <div class="subhead">候选事实</div>
                {render_list(facts, empty='本轮没有抽取到可持久化事实')}
                <div class="subhead">写入动作</div>
                <table class="grid-table">
                  <tr><th>事件</th><th>新记忆文本</th><th>旧记忆</th></tr>
                  {''.join(action_rows)}
                </table>
              </div>
            </details>
            """
        )
    return "".join(blocks) if blocks else "<div class='muted'>无 L3 记录。</div>"


def map_l2_by_session(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    mapped: Dict[str, Dict[str, Any]] = {}
    for item in data.get("all_l2_summaries", []):
        session_id = str(item.get("session_id", ""))
        mapped[session_id] = item
        if "_" in session_id:
            mapped[session_id.split("_")[-1]] = item
    return mapped


def render_session_panels(data: Dict[str, Any]) -> str:
    session_to_l2 = map_l2_by_session(data)
    sections: List[str] = []
    for session in data.get("session_results", []):
        session_label = session.get("session_label", "unknown")
        graph_data = session.get("end_result", {}).get("graph_data", {}) or {}
        treatment_report = session.get("end_result", {}).get("treatment_report", {}) or {}
        l2_item = session_to_l2.get(session_label) or session_to_l2.get(session.get("run_session_id", ""))
        total_stored = sum(int(item.get("l3_memory_result", {}).get("stored", 0)) for item in session.get("l3_records", []))
        relations = graph_data.get("relations", []) or []

        sections.append(
            f"""
            <section class="panel">
              <h2>{h(session_label)} 记忆落库展示</h2>
              <div class="session-meta">
                {badge(f"turns={len(session.get('l3_records', []))}", 'neutral')}
                {badge(f"L3写入={total_stored}", 'l3')}
                {badge(f"graph_relations={len(relations)}", 'graph')}
              </div>
              <div class="two-col">
                <div>
                  <div class="subhead">本次 L2 存储摘要</div>
                  {render_kv_table([
                      ('主题', (l2_item or {}).get('topic', '')),
                      ('摘要', (l2_item or {}).get('summary', '')),
                  ])}
                </div>
                <div>
                  <div class="subhead">本轮结束后的督导师输出</div>
                  {render_kv_table([
                      ('情绪趋势', get_first(treatment_report, ['emotion_trend'])),
                      ('治疗阶段', get_first(treatment_report, ['treatment_phase'])),
                      ('下一步重点', get_first(treatment_report, ['next_focus'])),
                  ])}
                </div>
              </div>
              <div class="subhead">session_graph 关系快照</div>
              {render_relation_table(relations)}
              <div class="subhead">L3 每轮写入过程</div>
              {render_l3_records(session.get('l3_records', []))}
            </section>
            """
        )
    return "".join(sections)


def render_treatment_report_panel(data: Dict[str, Any]) -> str:
    report = data.get("latest_treatment_report", {}) or {}
    progress = report.get("key_progress", []) or []
    return f"""
    <section class="panel">
      <h2>最新督导师报告</h2>
      <div class="two-col">
        <div>{render_kv_table([
            ('emotion_trend', get_first(report, ['emotion_trend'])),
            ('risk_note', get_first(report, ['risk_note'])),
            ('treatment_phase', get_first(report, ['treatment_phase'])),
            ('next_focus', get_first(report, ['next_focus'])),
        ])}</div>
        <div>
          <div class="subhead">key_progress</div>
          {render_list([str(item) for item in progress], empty='无关键进展')}
        </div>
      </div>
    </section>
    """


def build_filtered_context_text(result: Dict[str, Any], limit: int = 5) -> str:
    sections: List[str] = []
    l3_fragments = result.get("l3_fragments", []) or []
    graph_relations = result.get("graph_relations", []) or []

    if l3_fragments:
        sections.append("【L3 相关片段】")
        sections.extend(item.get("memory", "") for item in l3_fragments[:limit])
    if graph_relations:
        sections.append("【图关系召回】")
        sections.extend(
            f"- {item.get('source', '')} --{item.get('relationship', '')}--> {item.get('target', '')}"
            for item in graph_relations[:limit]
        )
    return "\n".join(sections).strip()


def render_retrieval_panel(data: Dict[str, Any]) -> str:
    cards: List[str] = []
    retrieval_examples = data.get("retrieval_examples", {}) or {}
    for query_name, result in retrieval_examples.items():
        l3_fragments = result.get("l3_fragments", []) or []
        graph_relations = result.get("graph_relations", []) or []
        cards.append(
            f"""
            <div class="retrieval-card">
              <div class="retrieval-title">{h(query_name)}</div>
              <div class="retrieval-meta">
                {badge(f"L3={len(l3_fragments)}", 'l3')}
                {badge(f"graph_relations={len(graph_relations)}", 'graph')}
              </div>
              <div class="subhead">按新口径拼接后的上下文</div>
              <div class="prelike">{h(build_filtered_context_text(result))}</div>
              <div class="subhead">图关系召回示例</div>
              {render_list([
                  f"{item.get('source', '')} --{item.get('relationship', '')}--> {item.get('target', '')}"
                  for item in graph_relations
              ], empty='无图关系召回')}
            </div>
            """
        )
    if not cards:
        cards.append("<div class='muted'>未找到检索样例。</div>")
    return f"""
    <section class="panel">
      <h2>后台检索展示</h2>
      <div class="card-grid">{''.join(cards)}</div>
    </section>
    """


def build_html(data: Dict[str, Any], results_path: Path, checkpoint_path: Optional[Path], summary_path: Optional[Path]) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Memory Demo - {h(data.get('user_id', 'unknown'))}</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdfa;
      --line: #d9ccb8;
      --ink: #2c2118;
      --muted: #7a6b5a;
      --accent: #af6f2b;
      --blue: #315c7c;
      --green: #2f6d5b;
      --red: #8b4a4a;
      --shadow: 0 14px 40px rgba(44, 33, 24, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #f7ddc0 0, transparent 26%),
        radial-gradient(circle at top right, #e8efe8 0, transparent 22%),
        linear-gradient(180deg, #f6f2eb 0%, #f1eadf 100%);
      line-height: 1.65;
    }}
    .wrap {{ width: min(1400px, calc(100vw - 40px)); margin: 32px auto 56px; }}
    .hero {{ background: linear-gradient(135deg, #2d2118 0%, #574130 100%); color: #fff7ef; border-radius: 28px; padding: 28px 32px; box-shadow: var(--shadow); }}
    .hero h1 {{ margin: 0 0 10px; font-size: 34px; line-height: 1.15; }}
    .hero p {{ margin: 0; color: #eadbca; }}
    .file-meta {{ margin-top: 14px; font-size: 13px; color: #d8c6b0; word-break: break-all; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin: 20px 0 28px; }}
    .metric-card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 22px; box-shadow: var(--shadow); }}
    .metric-card {{ padding: 18px 18px 16px; }}
    .metric-label {{ font-size: 13px; color: var(--muted); margin-bottom: 8px; }}
    .metric-value {{ font-size: 28px; font-weight: 700; }}
    .panel {{ padding: 22px 24px; margin-bottom: 20px; }}
    .panel h2 {{ margin: 0 0 16px; font-size: 24px; }}
    .subhead {{ margin: 16px 0 10px; font-weight: 700; color: var(--accent); }}
    .two-col {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .session-card, .retrieval-card {{ background: #fffaf4; border: 1px solid var(--line); border-radius: 18px; padding: 16px; }}
    .session-card-head {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; }}
    .session-name, .retrieval-title {{ font-weight: 700; font-size: 18px; }}
    .session-topic {{ margin-top: 10px; font-weight: 700; color: var(--blue); }}
    .session-summary {{ margin-top: 8px; color: var(--ink); white-space: pre-wrap; }}
    .badge {{ display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
    .badge-neutral {{ background: #f1e7d9; color: #6c4d2a; }}
    .badge-l2 {{ background: #e4edf6; color: #315c7c; }}
    .badge-l3 {{ background: #e5f2ee; color: #2f6d5b; }}
    .badge-graph {{ background: #f8eadf; color: #8a5722; }}
    .badge-green {{ background: #e5f2ee; color: #2f6d5b; }}
    .badge-blue {{ background: #e4edf6; color: #315c7c; }}
    .badge-red {{ background: #f7e5e4; color: #8b4a4a; }}
    .badge-gray {{ background: #ece7e2; color: #5c5147; }}
    .muted {{ color: var(--muted); font-size: 14px; }}
    .file-note {{ color: var(--muted); font-size: 13px; margin-bottom: 14px; word-break: break-all; }}
    .kv-table, .grid-table {{ width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 16px; border: 1px solid var(--line); background: #fffdfa; }}
    .kv-table th, .kv-table td, .grid-table th, .grid-table td {{ padding: 10px 12px; border-bottom: 1px solid #eadfce; vertical-align: top; text-align: left; }}
    .kv-table th, .grid-table th {{ background: #faf4ea; color: var(--muted); }}
    .kv-table th {{ width: 170px; }}
    .session-meta, .retrieval-meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
    .turn-box {{ border: 1px solid var(--line); border-radius: 16px; background: #fffcf8; margin-bottom: 12px; overflow: hidden; }}
    .turn-box summary {{ list-style: none; cursor: pointer; display: flex; justify-content: space-between; align-items: center; padding: 12px 14px; font-weight: 700; background: #fbf4ea; }}
    .turn-box summary::-webkit-details-marker {{ display: none; }}
    .turn-inner {{ padding: 14px; }}
    .turn-role {{ margin-bottom: 10px; white-space: pre-wrap; }}
    .prelike {{ white-space: pre-wrap; background: #fffaf4; border: 1px solid var(--line); border-radius: 14px; padding: 12px; font-size: 14px; min-height: 90px; }}
    .arch-grid {{ display: grid; grid-template-columns: 1.2fr auto 1fr auto 1fr auto 1fr; gap: 12px; align-items: stretch; }}
    .arch-grid-bottom {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top: 14px; }}
    .arch-box {{ border-radius: 20px; padding: 16px; border: 1px solid var(--line); min-height: 120px; }}
    .arch-title {{ font-weight: 800; margin-bottom: 8px; font-size: 17px; }}
    .arch-text {{ color: var(--muted); font-size: 14px; }}
    .arch-input {{ background: #fff7ef; }}
    .arch-l3 {{ background: #eef7f4; }}
    .arch-l2 {{ background: #eef4fb; }}
    .arch-l1 {{ background: #f8f0fb; }}
    .arch-graph {{ background: #fff3ea; }}
    .arch-report {{ background: #f5f1fb; }}
    .arch-retrieval {{ background: #eef7f0; }}
    .arch-arrow {{ display: grid; place-items: center; font-size: 28px; color: var(--accent); font-weight: 900; }}
    .explain-box {{ margin-top: 16px; background: #fff7ef; border: 1px solid var(--line); border-radius: 18px; padding: 16px 18px; }}
    .explain-title {{ font-weight: 800; margin-bottom: 8px; color: var(--accent); }}
    ol {{ margin: 0; padding-left: 22px; }}
    ul {{ margin: 8px 0 0; padding-left: 20px; }}
    li {{ margin-bottom: 6px; }}
    @media (max-width: 1080px) {{ .arch-grid {{ grid-template-columns: 1fr; }} .arch-arrow {{ transform: rotate(90deg); }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>心理咨询多层记忆演示面板</h1>
      <p>用于答辩和录屏展示。当前页面按你的最新展示口径整理：session_graph 只强调关系记忆，后台检索只展示 L3 与 graph_relations，不展示 L2 检索。</p>
      <div class="file-meta">
        结果文件: {h(results_path)}<br/>
        Checkpoint: {h(checkpoint_path) if checkpoint_path else '未找到'}<br/>
        L1 摘要文件: {h(summary_path) if summary_path else '未找到'}
      </div>
    </section>

    <section class="metrics">{build_overview_cards(data)}</section>
    {render_architecture_panel(data)}
    {render_l1_panel(data, summary_path)}
    {render_l2_panel(data)}
    {render_session_panels(data)}
    {render_treatment_report_panel(data)}
    {render_retrieval_panel(data)}
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="把 replay 结果渲染成适合答辩展示的 HTML 页面")
    parser.add_argument("--input", help="指定 replay_results.json 路径；不传则自动扫描最新文件")
    parser.add_argument("--output", help="输出 HTML 路径；不传则写到结果文件同目录")
    args = parser.parse_args()

    results_path = discover_results_file(args.input)
    data = load_json(results_path)
    checkpoint_path = discover_checkpoint_file(results_path)
    summary_path = discover_summary_file(str(data.get("user_id", "")))

    output_path = Path(args.output) if args.output else results_path.with_name(results_path.stem + "_demo.html")
    html_text = build_html(data, results_path, checkpoint_path, summary_path)
    write_text(output_path, html_text)

    resolved_output = output_path.resolve()
    print(f"已生成可视化页面: {resolved_output}")
    print(f"可点击绝对路径: {resolved_output}")
    print(f"可点击浏览器链接: {to_clickable_uri(output_path)}")
    print("当前展示口径: session_graph 只展示关系，检索只展示 L3 + graph_relations。")


if __name__ == "__main__":
    main()
