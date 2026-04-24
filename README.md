# CodeInsight-Agent

CodeInsight-Agent 是一个面向本地代码仓库的 **FastAPI + Web** 智能分析/改造助手。
它支持从纯问答到可执行 Agent 的完整链路：代码检索、文件读写、命令执行、测试回归、评测落盘与会话持久化。

## 优化后版本摘要

当前仓库已包含以下关键优化（对应最近迭代）：

- 上下文构建优化：预算约束 + 优先级去重，减少无效上下文占用。
- RAG 检索增强：`structured_v3` 分块、混合检索、增量索引刷新。
- 运行时缓存优化：LLM / Embedding / VectorStore 多级缓存，降低重复初始化开销。
- Agent 可靠性增强：支持 execute-and-verify 自动修复循环（最多重试 3 轮）。
- 可观测性增强：补充 reasoning / tool I/O / 复现场景信息，便于追踪问题。
- 兼容工具补齐：`code_search` / `open_file` / `find_symbol` / `run_code` / `run_tests`。
- 模型与向量后端扩展：新增 Ollama 推理与 embedding 后端支持。

## 评测结果（最近一次）

来源：`outputs/eval_result.json`（时间戳 `2026-04-05T15:31:55+08:00`）

- 总任务数：5
- 通过：5
- 失败：0
- 通过率：100%
- 平均任务耗时：36ms
- 检索命中率：100%
- 检索 MRR：0.375

## 核心能力

- 三种交互模式：`qa` / `workspace_qa` / `agentic`
- 会话管理：创建、更新、置顶、归档、删除，持久化到 `data/sessions/*.json`
- Web 流式响应：SSE 推送中间事件与最终答案
- 工作区能力：目录浏览、文件读取、可选写入
- Agent 执行：任务板、工具调用、可选命令执行与测试
- 评测可视化：读取 `outputs/eval_result.json` 并通过 `/eval/latest` 提供

## 项目结构

- `app/web/`：Web 接口、会话编排、流式响应、前端模板与静态资源
- `app/agent/`：Planner / Executor / Recovery / TaskBoard / Memory
- `app/rag/`：分块、向量索引、增量更新、检索器
- `app/tools/`：搜索、文件系统、补丁写入、命令执行、测试等工具
- `app/llm/`：多 provider LLM 统一封装（deepseek/openai/ollama）
- `scripts/`：索引构建、离线评测、产物清理
- `tests/`：核心单测与 Web API 测试

## 快速开始

### 1) 环境要求

- Python 3.10+
- 可用的 LLM API 或本地 Ollama

### 2) 安装依赖

```bash
pip install -r requirements.txt
```

### 3) 配置环境变量

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

最小 DeepSeek 配置示例：

```env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your_key
```

OpenAI 兼容配置示例：

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key
# OPENAI_BASE_URL=https://api.openai.com/v1
```

Ollama 配置示例：

```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
# OLLAMA_API_KEY=
```

Embedding 后端示例：

```env
# hash | sentence_transformers | openai | ollama
EMBEDDING_BACKEND=sentence_transformers
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### 4) 启动 Web

```bash
python -m app.web
```

默认地址：`http://127.0.0.1:8765`

自定义端口：

```bash
WEB_PORT=9000 python -m app.web
```

Windows PowerShell:

```powershell
$env:WEB_PORT=9000
python -m app.web
```

## 常用脚本

构建/刷新索引：

```bash
python scripts/build_index.py --workspace-root .
python scripts/build_index.py --workspace-root . --force-reindex
```

运行离线评测：

```bash
python scripts/run_eval.py
```

清理生成产物（先预览）：

```bash
python scripts/clear_state.py --dry-run
python scripts/clear_state.py --include-pytest-cache --include-pycache
python scripts/clear_state.py --remove-eval-result
```

运行测试：

```bash
pytest -q
```

## 主要 API

- `GET /sessions`
- `POST /sessions`
- `GET /sessions/{session_id}`
- `PATCH /sessions/{session_id}`
- `DELETE /sessions/{session_id}`
- `POST /sessions/{session_id}/messages`（支持 `?stream=true`）
- `POST /sessions/{session_id}/tests/run`
- `GET /sessions/{session_id}/workspace/tree`
- `GET /sessions/{session_id}/workspace/file`
- `PUT /sessions/{session_id}/workspace/file`
- `POST /system/pick-folder`
- `POST /system/pick-file`
- `GET /eval/latest`

## 关键配置项

- `LLM_PROVIDER`：`deepseek` / `openai` / `ollama`
- `LLM_MODEL`：模型名
- `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `OLLAMA_API_KEY`
- `OPENAI_BASE_URL` / `OLLAMA_BASE_URL`
- `EMBEDDING_BACKEND`：`hash` / `sentence_transformers` / `openai` / `ollama`
- `WEB_PORT`：Web 端口（默认 `8765`）
- `AGENT_ALLOW_WRITE`：CLI 模式允许写文件
- `AGENT_ALLOW_SHELL`：CLI 模式允许执行白名单命令

## 许可证

[MIT](./LICENSE)
