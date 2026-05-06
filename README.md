# MemAgent
### 基于双Agent与Mem0的心理咨询长期记忆系统
> 本科毕业设计｜多智能体协作｜三层记忆架构｜本地微调大模型｜私有化部署

[![GitHub](https://img.shields.io/badge/GitHub-Zoeywyt/MemAgent-blue)](https://github.com/Zoeywyt/MemAgent)

## 项目简介
MemAgent 是面向**长周期心理咨询场景**设计的智能对话记忆系统，基于 Mem0 记忆框架深度扩展，创新提出 **L1 长期画像 / L2 专业知识图谱 / L3 原文片段** 三层记忆结构，并搭配**双智能体分工协作**与**本地微调模型**，解决大模型在心理咨询中记忆碎片化、跨会话遗忘、共情不专业、缺乏结构化推理等问题。
系统支持私有化部署、记忆自动读写、混合检索、会话督导与可视化展示，可直接用于心理咨询 Agent 记忆增强与对话能力升级。

---

## 核心特性
- **三层记忆架构**
  - L1 长期画像：跨会话全局摘要、来访者状态与咨询进展
  - L2 专业图谱：心理问题、关系、行为模式结构化关系网络
  - L3 片段记忆：关键对话原文、细节语境与重要表述
- **双智能体协同**
  - EmpathyAgent（Qwen2.5-7B 微调）：负责共情交互、对话策略、主回复生成
  - SummaryAgent（Qwen2.5-3B 微调）：负责记忆压缩、摘要更新、图谱抽取
  - Supervisor：提供治疗方向督导与整体进展分析
- **本地模型微调**
  - SFT 监督微调：掌握咨询格式与专业干预策略
  - DPO 偏好对齐：提升真实性、专业性，抑制幻觉与无效安慰
- **混合检索引擎**
  融合向量检索、图关系检索、BM25 关键词检索，实现高精度记忆召回
- **完整工程闭环**
  会话管理、记忆更新、图谱写入、督导报告、HTML 可视化、自动化测试全流程

---

## 系统架构
1. **交互层**：用户输入 → EmpathyAgent 组织上下文 → 调用记忆
2. **记忆层**：Mem0 适配器统一管理 L1/L2/L3 存储、检索、更新
3. **智能体层**：SummaryAgent 生成摘要与图谱；Supervisor 输出督导结论
4. **存储层**：Chroma 向量库 + Kuzu 图数据库 + 本地 SQLite 记忆历史
5. **展示层**：HTML 可视化记忆结构、对话日志、督导报告、评估结果

---

## 技术栈
- 核心框架：Mem0
- 模型：Qwen2.5-7B-Instruct、Qwen2.5-3B-Instruct
- 训练方法：LoRA、SFT、DPO
- 存储：Chroma（向量）、Kuzu（图）、SQLite（记忆历史）
- 评估：BLEU、ROUGE、BERTScore、CPsyCounR 四维评估体系
- 工程：Python、dotenv 配置、自动化测试、HTML 可视化

---

## 项目结构
```

## Local model mode

You can now choose a backend per component:

- `gpt`: use the existing OpenAI-compatible API
- `qwen3b`: use `models/Qwen2.5-3B-Instruct` + `models/Qwen2.5-3B-Instruct-Lora`

Example mixed setup:

```env
MA_AGENT_MODEL_BACKEND=qwen3b
MA_SUMMARY_MODEL_BACKEND=qwen3b
MA_SUPERVISOR_MODEL_BACKEND=gpt
MA_GRAPH_MODEL_BACKEND=gpt
MEM0_L3_LLM_MODEL_BACKEND=gpt
```

If you want a custom local path instead of the built-in presets, you can still use:

- `*_MODEL_MODE=local`
- `*_LOCAL_MODEL_PATH`
- `*_LOCAL_BASE_MODEL_PATH`

## Gradio UI

`main.py` now launches a Gradio web app by default.

Features:

- user login / session start
- per-component model backend selection
- live chat UI
- memory + report dashboard rendered in the same style as `render_memory_demo.py`
- report export as zip/html/json/txt
- direct history import for replay/report generation using the `test_case.py` replay format

Run:

```bash
python main.py
```

Optional:

```env
MA_UI_MODE=gradio
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860
```
