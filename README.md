# CodeInsight-Agent

一个基于 FastAPI 的本地代码助手 Web 应用，提供会话管理、聊天、工作区文件树、代码编辑、流式响应，以及面向当前项目的 Agent 执行能力。

当前仓库主要包含：

- Web UI：会话列表、聊天区、文件树、编辑器、测试入口
- Agent 执行链路：Planner / Executor / Memory / Workspace 工具
- 面向当前工作区的读写、搜索、测试触发能力
- 会话持久化、索引构建、检索评测与清理脚本

如果你只是想尽快跑起来，先看下面的“快速开始”。

## 快速开始

### 1. 环境要求

- Python `3.10+`
- Windows / macOS / Linux
- 可用的 LLM API Key

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 `.env`

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

最少需要配置一组可用模型，例如使用 DeepSeek：

```env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=你的_key
```

如果你想使用 OpenAI 兼容接口，可以改成：

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=你的_key
# OPENAI_BASE_URL=https://api.openai.com/v1
```

### 4. 启动 Web

推荐直接运行：

```bash
python -m app.web
```

默认地址：

`http://127.0.0.1:8765`

如果你想自定义端口：

```bash
WEB_PORT=9000 python -m app.web
```

Windows PowerShell：

```powershell
$env:WEB_PORT=9000
python -m app.web
```

也可以用 `uvicorn`：

```bash
uvicorn app.web.main:app --reload --port 8765
```

## Web 怎么用

### 模式 1：问答模式

适合只想提问，不需要访问本地代码。

1. 打开页面后新建会话
2. `workspace_root` 留空
3. 直接在右侧输入问题并发送

这种模式不会依赖本地工作区。

### 模式 2：任务模式

适合让 Agent 读取、修改、测试当前项目。

1. 新建会话
2. 在会话设置里填写 `workspace_root`
3. 需要改代码时，打开 `allow_write`
4. 需要跑命令或测试时，打开 `allow_shell`
5. 如果要使用“运行测试”按钮，填写 `test_command`
6. 在右侧输入任务，例如“修复登录接口并补最小测试”

常见设置说明：

- `workspace_root`：当前会话绑定的项目目录，支持绝对路径，也支持相对于仓库根目录的路径
- `allow_write`：允许 Agent 修改工作区内文件
- `allow_shell`：允许 Agent 执行命令
- `test_command`：例如 `python -m pytest -q`
- `auto_run_tests`：检测到写入后自动运行测试
- `max_turns`：单次任务最多执行的 agent 回合数

受控命令默认允许一组低风险只读前缀：
- `git status`
- `git diff`
- `rg`
- `pytest`
- `python -m pytest`
- `python -m compileall`
- `ruff check`

### 文件树 / 编辑器 / 会话区

- 左侧：当前工作区文件树
- 中间：代码编辑器，支持多标签
- 右侧：会话列表、聊天区、发送/停止按钮

会话数据默认保存在：

`data/sessions/<session_id>.json`

## 前端资源结构

当前 Web 前端已经从单个超长模板拆成模板和静态资源，方便维护：

- `app/web/templates/index.html`：页面骨架和 DOM 结构
- `app/web/static/web/index.css`：页面样式
- `app/web/static/web/index.js`：前端交互、布局拖拽、会话与编辑器逻辑
- `app/web/main.py`：挂载 `/static`，主页会自动加载上述 CSS / JS

如果要调整前端，通常只需要修改 `index.css` 或 `index.js`，而不是继续在 `index.html` 里维护大段内联代码。

## 首次使用建议

如果你准备在真实项目里长期使用任务模式，建议先构建索引：

```bash
python scripts/build_index.py --workspace-root .
```

强制重建：

```bash
python scripts/build_index.py --workspace-root . --force-reindex
```

如果你只是用问答模式聊天，可以先不做这一步。

## 常用命令

### 启动 Web

```bash
python -m app.web
```

### 构建索引

```bash
python scripts/build_index.py --workspace-root .
```

### 运行评测

```bash
python scripts/run_eval.py
```

评测结果会写到：

`outputs/eval_result.json`

Web 页面里的“最近评测”和 `/eval/latest` 会读取这个文件，并展示：
- `success_rate / task_completion_rate`
- `retrieval_hit_rate / retrieval_mrr`
- `avg_duration_ms / recovery_trigger_rate`

如果你只是想快速验证评测流程，或者本地 embedding 首次加载较慢，可以先在 `.env` 里设置：

```env
EMBEDDING_BACKEND=hash
```

### 清理产物

先看会删什么：

```bash
python scripts/clear_state.py --dry-run
```

常见清理：

```bash
python scripts/clear_state.py --include-pytest-cache --include-pycache
python scripts/clear_state.py --remove-eval-result
```

### 跑测试

```bash
pytest -q
```

## 目录说明

- `app/web/main.py`：FastAPI 入口，负责挂载静态资源
- `app/web/templates/index.html`：前端页面骨架
- `app/web/static/web/index.css`：前端样式
- `app/web/static/web/index.js`：前端交互逻辑
- `app/web/service.py`：Web 服务层
- `app/web/session_store.py`：会话持久化
- `scripts/build_index.py`：构建索引
- `scripts/run_eval.py`：运行最小评测
- `scripts/clear_state.py`：清理产物

## 常见问题

### 1. 启动后打不开页面

先看终端是否报错，然后确认端口是否被占用。默认端口是 `8765`。

### 2. 任务模式提示 `workspace_root` 无效

说明你填写的目录不存在，或者不是目录。请改成真实存在的本地项目路径。

### 3. 第一次构建索引很慢

这是正常的。项目较大时，首次索引会花一些时间；后续会复用持久化结果。

### 4. 不想依赖本地 embedding 模型

可以在 `.env` 里设置：

```env
EMBEDDING_BACKEND=hash
```

这适合快速跑通流程，但检索质量会差一些。

### 5. Web 里为什么不能直接改文件或跑测试

因为这是会话级权限控制：

- 没开 `allow_write`，就不能保存文件
- 没开 `allow_shell`，就不能执行测试命令

## 一个最短上手流程

```bash
pip install -r requirements.txt
cp .env.example .env
python -m app.web
```

然后：

1. 打开 `http://127.0.0.1:8765`
2. 新建会话
3. 填 `workspace_root`
4. 按需打开 `allow_write` / `allow_shell`
5. 输入任务并发送
