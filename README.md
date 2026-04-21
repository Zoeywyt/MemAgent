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
MemAgent/
├── agents/                # 智能体模块
│   ├── empathy_agent.py   # 共情交互主Agent
│   ├── summary_agent.py   # 摘要与记忆压缩Agent
│   └── supervisor.py      # 督导分析模块
├── memory/                # 记忆核心模块
│   ├── mem0_adapter.py    # Mem0三层记忆适配器
│   └── graph_extractor.py # 心理关系图谱抽取
├── prompts/               # 提示词模板
├── train/                 # SFT/DPO/摘要微调代码
├── utils/                 # 工具函数
├── main.py                # 系统入口
├── test_case.py           # 案例回放与自动测试
├── render_memory_demo.py  # 记忆可视化
├── requirements.txt       # 依赖清单
└── .env                   # 环境配置
```

---

## 快速开始
### 1. 克隆项目
```bash
git clone https://github.com/Zoeywyt/MemAgent.git
cd MemAgent
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量（.env）
```env
OPENAI_BASE_URL=your_base_url
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=your_model_name

GRAPH_LLM_BASE_URL=your_graph_llm_url
GRAPH_LLM_API_KEY=your_api_key
GRAPH_LLM_MODEL=your_model_name

MEM0_ENABLE_GRAPH=True
CHROMA_DB_PATH=./chroma_db
KUZU_DB_PATH=./mem0_kuzu_db
MEM0_HISTORY_DB=./mem0_history.db
```

### 4. 运行案例测试
```bash
python test_case.py
```

### 5. 查看可视化结果
运行后自动生成 HTML 页面，展示：
- 多轮对话日志
- L1/L2/L3 记忆内容
- 心理关系图谱
- 督导分析报告
- 系统执行日志
- 
<img width="1472" height="922" alt="image" src="https://github.com/user-attachments/assets/22b658d4-1132-45d7-a9da-22b6527583ec" />
<img width="1494" height="1080" alt="image" src="https://github.com/user-attachments/assets/89e03938-8718-4bf2-a105-2990e8948195" />
<img width="1241" height="1167" alt="image" src="https://github.com/user-attachments/assets/75a3f92f-368b-4366-aaa1-6f37698c5e1e" />

---

## 模型与数据说明
- 数据集：基于 CPsyCounR 真实心理咨询报告构建，总计 **19,856 条 DPO 样本、7,851 条 SFT 样本**
- 主模型：Qwen2.5-7B-Instruct（SFT + DPO 对齐）
- 摘要模型：Qwen2.5-3B-Instruct（专项微调）
- 训练目标：专业咨询策略、情感反映、认知重构、安全抑制、长期记忆对齐

---

## 评估结果
| 指标         | Baseline | SFT     | SFT+DPO |
|--------------|----------|---------|---------|
| BLEU-4       | 11.2     | 15.8    | 15.2    |
| ROUGE-L      | 22.4     | 27.0    | 26.5    |
| BERTScore    | 0.823    | 0.883   | 0.887   |

系统在**专业性、真实性、安全性、全面性**四维评估中显著优于基线，有效提升长周期咨询质量。

---

## 毕业设计创新点
1. 在 Mem0 基础上创新**三层心理咨询记忆架构**，实现长时记忆结构化管理
2. 双智能体分工协作，分离交互与记忆，提升系统稳定性与可扩展性
3. 基于真实临床数据进行 SFT+DPO 对齐，更贴合心理咨询专业要求
4. 端到端私有化部署，不依赖第三方 API，保护隐私与数据安全
5. 完整可视化、自动化测试、消融实验设计，成果可复现可验证

---
