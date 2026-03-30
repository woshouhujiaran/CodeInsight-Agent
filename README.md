# CodeInsight-Agent

精简后的仓库仅保留 Web 使用路径：

- FastAPI Web 服务与前端界面
- Agent / Planner / Executor / Memory 核心逻辑
- Web 会话管理、聊天与任务模式
- Web 会话内可触发的测试运行能力

CLI 入口、离线评估脚本和对应测试已移除。

## 安装

```bash
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，并至少配置可用的 LLM 凭据，例如：

- `DEEPSEEK_API_KEY=...`
- 可选：`LLM_PROVIDER=deepseek`
- 可选：`LLM_MODEL=deepseek-chat`

## 启动 Web

```bash
python -m app.web
```

或：

```bash
uvicorn app.web.main:app --reload --port 8765
```

默认地址：`http://127.0.0.1:8765`

## Web 能力

- 会话列表与本地持久化，默认保存在 `data/sessions/<session_id>.json`
- Web 聊天与流式返回
- 任务模式与任务看板
- 针对当前会话工作区的读写、搜索、分析和测试能力
- 基于会话配置的项目测试运行

## 工作区与权限

- `workspace_root` 支持绝对路径
- 也支持相对于仓库根目录的相对路径
- Web 后端会解析为绝对路径，并要求目标目录真实存在

会话设置中的关键开关：

- `allow_write`
  - 允许 Agent 在当前工作区内修改文件
- `allow_shell`
  - 允许 Agent 在当前工作区内执行白名单命令
- `test_command`
  - Web “运行测试”按钮与自动测试所使用的命令
- `auto_run_tests`
  - 任务模式中检测到成功写入后自动运行测试
- `max_turns`
  - 单个任务的 agentic 回合上限

环境变量：

- `WEB_PORT`
  - `python -m app.web` 的默认端口，默认为 `8765`

## 测试

仓库保留了与 Web 主流程、Agent 核心逻辑和 Web 可调用工具相关的测试。运行全部保留测试：

```bash
pytest
```
