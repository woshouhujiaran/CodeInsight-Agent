# CodeInsight-Agent

CodeInsight-Agent 是一个基于 FastAPI 的本地代码助手 Web 应用，用来围绕当前仓库做问答、工作区分析和 Agent 执行。

它支持三种主要工作方式：

1. `QA`：纯问答，不读取本地仓库，也不修改文件。
2. `workspace_qa`：读取当前工作区中的文件做解释和总结，但默认只读。
3. `agentic`：以任务板方式执行检索、分析、修改、测试等动作。

应用会把会话持久化到 `data/sessions/*.json`，并可为每个工作区构建独立的 RAG 索引。

## 主要功能

1. 会话管理：新建、查看、重命名、置顶、归档、删除会话。
2. 工作区浏览：查看目录树、读取文件、在允许写入时保存文件。
3. 模式切换：根据用户输入自动判断是 QA、workspace_qa 还是 agentic。
4. Agent 执行：在允许时使用检索、文件读取、目录浏览、补丁写入、命令执行和测试工具。
5. 流式输出：Web 端支持 SSE 流式返回中间过程。
6. 评测查看：Web 页面可读取 `outputs/eval_result.json` 中的最新评测结果。
7. 本地路径选择：支持通过系统文件选择器选取文件或文件夹。

## 项目结构

1. `app/web/main.py`：FastAPI 入口和 HTTP 路由。
2. `app/web/service.py`：Web 层核心业务，负责会话、模式路由和工作区操作。
3. `app/web/chat_components.py`：模式判定、安全拦截、澄清提示、任务板响应渲染。
4. `app/web/session_store.py`：会话持久化与规范化。
5. `app/runtime.py`：LLM、embedding、RAG、Agent 和工具注册。
6. `app/agent/`：Planner、Executor、Memory、TaskBoard、Recovery 等 Agent 逻辑。
7. `app/rag/`：索引构建、向量存储、检索和分块。
8. `app/tools/`：文件读写、搜索、命令执行、测试等工具。
9. `scripts/build_index.py`：构建或刷新工作区索引。
10. `scripts/run_eval.py`：运行离线评测。
11. `scripts/clear_state.py`：清理生成产物。

## 快速开始

### 1. 环境要求

1. Python `3.10+`
2. Windows、macOS 或 Linux
3. 可用的 LLM API Key

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 `.env`

复制示例文件：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

至少要配置一个可用的模型提供方。例如 DeepSeek：

```env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your_key
```

如果使用 OpenAI 兼容接口：

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key
# OPENAI_BASE_URL=https://api.openai.com/v1
```

如果使用 Ollama 本地模型：
```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
# OLLAMA_API_KEY=
```

如果想降低依赖、快速跑通，可以把 embedding 后端改成 hash，但检索质量会下降：

```env
EMBEDDING_BACKEND=hash
```

### 4. 启动 Web

推荐直接运行：

```bash
python -m app.web
```

默认地址：

`http://127.0.0.1:8765`

也可以指定端口：

```bash
WEB_PORT=9000 python -m app.web
```

Windows PowerShell：

```powershell
$env:WEB_PORT=9000
python -m app.web
```

或者使用 uvicorn：

```bash
uvicorn app.web.main:app --reload --port 8765
```

## Web 使用方式

### 1. 纯问答

适合只想聊天、解释概念、对比原理，不需要访问本地代码的场景。

1. 新建会话。
2. `workspace_root` 可以留空。
3. 直接提问。

### 2. 工作区问答

适合解释当前仓库里的文件、模块和入口，不需要修改代码。

1. 新建会话。
2. 先设置有效的 `workspace_root`。
3. 提问时尽量明确文件、目录、模块或入口。
4. 默认只读，不会自动改文件或跑测试。

### 3. Agent 执行

适合分析、修改和验证当前项目。

1. 新建会话。
2. 设置真实存在的 `workspace_root`。
3. 需要改文件时打开 `allow_write`。
4. 需要运行命令或测试时打开 `allow_shell`。
5. 如果要让系统自动跑测试，再填写 `test_command`。

建议优先在会话里明确这些设置：

1. `workspace_root`：会话绑定的工作区目录。
2. `allow_write`：允许写文件。
3. `allow_shell`：允许执行允许列表中的命令。
4. `test_command`：例如 `python -m pytest -q`。
5. `auto_run_tests`：检测到写入后自动跑测试。
6. `max_turns`：单次 Agent 执行的最大轮次。

### 4. 会话数据

会话会保存到：

`data/sessions/<session_id>.json`

## RAG 索引

如果你打算在真实项目里长期使用 Agent，建议先构建索引：

```bash
python scripts/build_index.py --workspace-root .
```

强制重建：

```bash
python scripts/build_index.py --workspace-root . --force-reindex
```

索引会放在 `data/index/<digest>` 下，按工作区隔离。

## 常用脚本

### 构建索引

```bash
python scripts/build_index.py --workspace-root .
```

### 运行评测

```bash
python scripts/run_eval.py
```

评测结果默认写到：

`outputs/eval_result.json`

Web 页面中的“最近评测”和 `/eval/latest` 会读取这个文件。

### 清理产物

先预览会删什么：

```bash
python scripts/clear_state.py --dry-run
```

清理常见生成物：

```bash
python scripts/clear_state.py --include-pytest-cache --include-pycache
python scripts/clear_state.py --remove-eval-result
```

### 运行测试

```bash
pytest -q
```

## 常见接口

1. `GET /sessions`：列出会话。
2. `POST /sessions`：创建会话。
3. `GET /sessions/{session_id}`：读取会话快照。
4. `PATCH /sessions/{session_id}`：更新会话设置。
5. `DELETE /sessions/{session_id}`：删除会话。
6. `POST /sessions/{session_id}/messages`：发送消息，支持 `?stream=true`。
7. `POST /sessions/{session_id}/tests/run`：按会话配置运行测试。
8. `GET /sessions/{session_id}/workspace/tree`：读取工作区目录树。
9. `GET /sessions/{session_id}/workspace/file`：读取工作区文件。
10. `PUT /sessions/{session_id}/workspace/file`：写入工作区文件。
11. `POST /system/pick-folder`：打开文件夹选择器。
12. `POST /system/pick-file`：打开文件选择器。
13. `GET /eval/latest`：读取最新评测结果。

## 配置说明

### LLM

1. `LLM_PROVIDER`：`deepseek`、`openai` 或 `ollama`。
2. `LLM_MODEL`：模型名。
3. `DEEPSEEK_API_KEY`：DeepSeek Key。
4. `OPENAI_API_KEY`：OpenAI 或兼容接口 Key。
5. `OPENAI_BASE_URL`：自定义兼容接口地址，可选。
6. `OLLAMA_BASE_URL`：Ollama OpenAI 兼容地址（默认 `http://127.0.0.1:11434/v1`）。

### Embedding

1. `EMBEDDING_BACKEND=sentence_transformers`：默认推荐，检索质量较好。
2. `EMBEDDING_BACKEND=openai`：使用 OpenAI 兼容 embeddings 接口。
3. `EMBEDDING_BACKEND=hash`：无需额外模型，适合快速验证。
4. `EMBEDDING_BACKEND=ollama`：使用 Ollama `/api/embed` 本地 embedding。

### Web

1. `WEB_PORT`：Web 端口，默认 `8765`。

### Agent 权限

1. `allow_write`：允许文件写入工具。
2. `allow_shell`：允许命令执行工具，但只会注册允许的命令。
3. `test_command`：当 `allow_shell` 开启时，测试命令可被允许执行。

## 备注

1. 读取仓库源码前，建议先构建索引，这样检索和 Agent 体验会稳定很多。
2. 如果只是问答，不需要填写 `workspace_root`。
3. 如果提问涉及当前项目的文件、目录或入口，尽量先设置 `workspace_root`。
4. 如果要改文件或跑测试，必须明确打开对应权限。
