# CodeInsight-Agent

精简后的仓库仅保留 Web 使用路径，包含以下能力：
- FastAPI Web 服务与前端界面
- Agent / Planner / Executor / Memory 核心逻辑
- Web 会话管理、聊天与任务模式
- Web 会话内可触发的测试运行能力
- 可供 Web 场景直接使用的索引构建、评估和清理脚本

## 安装

```bash
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，并至少配置可用的 LLM 凭据，例如：

- `DEEPSEEK_API_KEY=...`
- 可选：`LLM_PROVIDER=deepseek`
- 可选：`LLM_MODEL=deepseek-chat`

如需稳定、无外部依赖的索引构建，可选设置：

- `EMBEDDING_BACKEND=hash`

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
- 最近一次评估结果展示，数据来自 `outputs/eval_result.json`

## 工作区与权限

- `workspace_root` 支持绝对路径
- 也支持相对于仓库根目录的相对路径
- Web 后端会解析为绝对路径，并要求目标目录真实存在

会话设置中的关键开关：

- `allow_write`：允许 Agent 在当前工作区内修改文件
- `allow_shell`：允许 Agent 在当前工作区内执行白名单命令
- `test_command`：Web “运行测试”按钮与自动测试使用的命令
- `auto_run_tests`：任务模式中检测到成功写入后自动运行测试
- `max_turns`：单个任务的 agentic 回合上限

环境变量：

- `WEB_PORT`：`python -m app.web` 的默认端口，默认为 `8765`

## Web 用户工作流

1. 启动 Web

Windows PowerShell：

```powershell
python -m app.web
```

通用命令：

```bash
python -m app.web
```

2. 首次使用前构建索引

Windows PowerShell：

```powershell
python .\scripts\build_index.py --workspace-root .
```

通用命令：

```bash
python scripts/build_index.py --workspace-root .
```

可选强制重建：

```bash
python scripts/build_index.py --workspace-root . --force-reindex
```

3. 使用过程中定期评估

Windows PowerShell：

```powershell
python .\scripts\run_eval.py
```

通用命令：

```bash
python scripts/run_eval.py
```

脚本会生成 `outputs/eval_result.json`，Web 端的“最近评测”卡片和 `/eval/latest` API 会直接读取该文件。

如果存在 `eval/tasks.json`，评估脚本会优先读取该文件；否则使用内置的最小任务集。

4. 清理产物

Windows PowerShell：

```powershell
python .\scripts\clear_state.py --dry-run
```

通用命令：

```bash
python scripts/clear_state.py --dry-run
```

常见选项：

```bash
python scripts/clear_state.py --include-pytest-cache --include-pycache
python scripts/clear_state.py --remove-eval-result
```

默认只清理 `outputs/generated_test_*.py`，并保留 `outputs/eval_result.json`。

## 脚本说明

- `scripts/build_index.py`：构建或加载当前工作区索引，输出 `index_dir`、`status` 和 `snapshot`
- `scripts/run_eval.py`：运行最小评估任务，并写入 `outputs/eval_result.json`
- `scripts/clear_state.py`：清理生成测试文件与可选缓存，支持 `--dry-run`

## 测试

仓库保留了与 Web 主流程、Agent 核心逻辑和 Web 可调用工具相关的测试。运行全部测试：

```bash
pytest
```
