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

## 常用参数

- `--codebase-dir` 指定入库目录（默认 `data/codebase`）
- `--top-k` 检索返回条数（默认 5）
- `--query` 单轮提问