# Evaluate

这个目录是 MemAgent 的评估入口和迁移代码。新环境里只需要同时准备：

1. `CPsyCoun` 仓库
2. `MoodBench` 仓库
3. `MemAgent/evaluate` 目录

推荐目录结构：

```text
MemAgent/
  CPsyCoun/
  MoodBench/
  evaluate/
  models/
```

兼容目录结构：

```text
workspace/
  CPsyCoun/
  MoodBench/
  MemAgent/
    evaluate/
    models/
```

## 评估范围

- MoodBench：复用官方 `MoodBench` 仓库里的 PQAEF 任务、测试配置和加权统计脚本。
- CPsyCoun：复用官方 `CPsyCoun/CPsyCounE` 数据；评分标准保留 turn-based 四维口径：`Comprehensiveness`、`Professionalism`、`Authenticity`、`Safety`。
- 不再使用 FEEL、ESC-Eval、ESC-RANK、ESC-Role 或 InternLM2。
- 不再依赖 `CPsyCoun/Code` 里的个人脚本；CPsyCoun 评分执行逻辑已迁移到 `evaluate/cpsycoun_evaluator.py`。

## 快速验证

在 `MemAgent` 目录下运行：

```powershell
& ".venv\Scripts\python.exe" -m evaluate.run_eval --stage cpsycoun --models base,sft,dpo --limit 2 --cpsycoun-dry-run --cpsycoun-max-workers 1
```

MoodBench 小样本：

```powershell
& ".venv\Scripts\python.exe" -m evaluate.run_eval --stage moodbench --models base,sft,dpo --datasets PQEmotion4 --limit 2
```

完整流程：

```powershell
& ".venv\Scripts\python.exe" -m evaluate.run_eval --stage all --models base,sft,dpo --base-model "BASE_MODEL_PATH" --sft-adapter ".\models\output_sft_7b" --dpo-adapter ".\models\output_dpo_7b" --moodbench-root ".\MoodBench" --cpsycoun-root ".\CPsyCoun"
```

## 常用参数

- `--base-model`：基础模型路径，默认 `MemAgent/models/Qwen2.5-7B-Instruct`
- `--sft-adapter`：SFT LoRA 路径，默认 `MemAgent/models/output_sft_7b`
- `--dpo-adapter`：DPO LoRA 路径，默认 `MemAgent/models/output_dpo_7b`
- `--moodbench-root`：MoodBench 仓库路径；默认优先查找 `MemAgent/MoodBench`，其次查找 `workspace/MoodBench`
- `--cpsycoun-root`：CPsyCoun 仓库路径；默认优先查找 `MemAgent/CPsyCoun`，其次查找 `workspace/CPsyCoun`
- `--cpsycoun-use-existing-input --cpsycoun-input <json>`：使用已经生成好的多轮对话 JSON 直接评分

## 输出

- MoodBench 原始结果：`MoodBench/output/test/<model>/<dataset>/statistical_analysis/result_stats.json`
- MoodBench 汇总副本：`MemAgent/evaluate/results/moodbench_scores.json`
- CPsyCoun 生成对话：`MemAgent/evaluate/output/cpsycoun/generated_inputs/<run_id>/<model>_cpsycoun_generated.json`
- CPsyCoun 四维评分：`MemAgent/evaluate/output/cpsycoun/<model>/<run_id>/evaluation_gpt54.csv`
- CPsyCoun 评分副本：`MemAgent/evaluate/results/cpsycoun_results/`
