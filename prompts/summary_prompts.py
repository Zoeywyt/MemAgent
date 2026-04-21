L2_SESSION_SUMMARY_PROMPT = """你是一个心理咨询记录提取Agent。请根据以下【单次会话的完整对话历史】，生成一个会话级摘要。

【输出格式要求】
只输出一个合法的JSON对象，键名为：topic、background、summary。

- topic: 本次会话的核心主题（1-2个短语）
- background: 用户关键信息描述，含情绪变化轨迹、重要事件、应对资源等
- summary: 会话过程与成果总结，含咨询师策略、用户反应、结论、作业、风险评估

【对话历史】
{conversation_history}
"""

INTEGRATED_SUMMARY_INSTRUCTION = """你是一个心理咨询记录整合Agent。请根据现有全局摘要和最新一次会话的摘要，生成更新后的完整全局摘要。
只输出一个合法的JSON对象，键名为：主题、背景、会话总结。

【现有全局摘要】
{existing_summary}

【最新会话摘要】
{new_session_summary}

要求：
- 主题：若出现新主题，追加到原有主题后（用分号分隔）。
- 背景：融合新旧信息，保持简洁完整。
- 会话总结：概括全局进展，体现最新变化。
"""
