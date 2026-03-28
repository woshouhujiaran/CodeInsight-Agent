# CodeInsight-Agent

最小可运行版本已包含：
- Agent 主循环（Planner / Executor / Memory）
- 工具系统（search / analyze / optimize / test）
- RAG 入库与 FAISS 检索
- Sandbox 测试执行（subprocess + timeout）

## 快速开始

1) 安装依赖

```bash
pip install -r requirements.txt
```

1.1) 配置 API Key（DeepSeek）

- 复制 `.env.example` 为 `.env`
- 在 `.env` 填写：
  - `DEEPSEEK_API_KEY=你的key`
  - 可选：`LLM_PROVIDER=deepseek`、`LLM_MODEL=deepseek-chat`

2) 准备待分析代码

将代码放到 `data/codebase/`（首次运行会自动入库）

3) 单轮模式

```bash
python -m app.main --query "分析这个项目的核心模块并给优化建议"
```

4) 多轮交互模式

```bash
python -m app.main
```

**运行模式说明**：默认使用 **Planner** 一次性生成工具计划并执行后再总结；加 **`--agentic`** 时改为 **`run_agentic`** 多轮 JSON 工具循环（可用 **`--max-turns`** 限制每轮用户输入下的大模型步数，默认 8）。

## 常用参数

- `--codebase-dir` 指定入库目录（默认 `data/codebase`）
- `--top-k` 检索返回条数（默认 5）
- `--query` 单轮提问
- `--agentic` / `--max-turns`：见上文「运行模式说明」

## E2E 评测基线

已提供 `>=20` 条任务评测集：`eval/tasks.json`（覆盖 analysis / optimize / test）。

一键运行评测：

```bash
python scripts/run_eval.py --tasks eval/tasks.json --output outputs/eval_result.json
```

可选参数：
- `--codebase-dir`：评测时使用的代码库目录（默认 `data/codebase`）
- `--top-k`：检索 top-k（默认 `5`）
- `--force-reindex`：评测前强制重建索引
- `--agentic` / `--max-turns`：评测走 **`run_agentic`**（默认仍为 **`agent.run`** / Planner）；单条任务可在 `eval/tasks.json` 里加 **`"agentic": true|false`** 覆盖全局 CLI

评测输出：
- 终端打印汇总指标：
  - `success_rate`
  - `avg_duration_ms`
  - `recovery_trigger_rate`
- 详细结果保存为 JSON：`outputs/eval_result.json`