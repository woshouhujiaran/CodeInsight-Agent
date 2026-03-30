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

## Web 模式

启动 Web：

```bash
python -m app.web
```

或：

```bash
uvicorn app.web.main:app --reload --port 8765
```

默认地址：`http://127.0.0.1:8765`

Web 页面提供：
- 左侧会话列表
- 中间多轮对话与流式回复
- 右侧任务看板、测试结果、环境开关
- 本地会话持久化：`data/sessions/<session_id>.json`

工作区策略：
- 支持绝对路径
- 也支持相对于当前仓库根目录的相对路径
- 后端会统一解析为绝对路径，并要求目标目录真实存在

## 环境变量与风险说明

- `AGENT_ALLOW_WRITE=1`
  - 允许 CLI Agent 注册 `apply_patch_tool` 与 `write_file_tool`
  - 有写盘风险，默认关闭
- `AGENT_ALLOW_SHELL=1`
  - 允许 CLI Agent 注册 `run_command_tool`
  - 有命令执行风险，默认关闭
- `WEB_PORT`
  - `python -m app.web` 的默认端口，默认 `8765`

Web 会话也有对应的“允许写盘 / 允许命令执行”开关；只有显式开启后，Agent 才会在用户指定工作区内修改文件或运行测试命令。

## 常用参数

- `--codebase-dir` 指定入库目录（默认 `data/codebase`）
- `--top-k` 检索返回条数（默认 5）
- `--query` 单轮提问
- `--agentic` / `--max-turns`：见上文「运行模式说明」

CLI 与 Web 的区别：
- CLI 直接绑定一个代码库目录运行
- Web 以会话为单位保存 `workspace_root`、消息历史、任务状态和最近一次测试结果
- Web 端测试运行来自会话配置 `test_command`，评测仍通过 `scripts/run_eval.py` 执行

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

Web 端只读展示最近一次评测结果，不负责触发批量评测；如需刷新该结果，请重新运行 `scripts/run_eval.py`。
