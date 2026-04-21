# multi_agent

这是一个参考 `code/src` 目录组织的多 agent 心理咨询示例项目。

## 目录结构

- `agents/`：共情、总结、督导师三个主要 agent
- `memory/`：Mem0 记忆适配器与图谱提取
- `prompts/`：本地 prompt 模板
- `utils/`：OpenAI 兼容 SSE 调用客户端

## 当前实现

- `empathy_agent.py`：使用 OpenAI 兼容接口生成回复，并在回复开头标注 `[策略]`
- `summary_agent.py`、`supervisor.py`：使用本地微调模型 `code/src/models/Qwen2.5-3B-Instruct-Lora`
- `memory/mem0_adapter.py`：已对齐 `code/src`，使用真实 Mem0 + Chroma 向量库 + Kuzu 图库存储
- `memory/graph_extractor.py`：已接入外部 OpenAI 兼容 LLM，用于提取 Mem0 可写入的实体与关系
- `end_session()`：结束会话时会写入图谱记忆、保存 L2 摘要、更新 L1 摘要，并生成治疗进展报告

## 运行前配置

需要在 `.env` 中至少提供这些变量：

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

如果需要图谱提取，还需要：

- `GRAPH_LLM_BASE_URL`
- `GRAPH_LLM_API_KEY`
- `GRAPH_LLM_MODEL`

可选的 Mem0 / 向量库配置：

- `MEM0_ENABLE_GRAPH`
- `KUZU_DB_PATH`
- `CHROMA_DB_PATH`
- `MEM0_HISTORY_DB`
- `EMBEDDING_MODEL_NAME`
- `MEM0_COLLECTION_NAME`

## 运行

```bash
python -m multi_agent.main
```
