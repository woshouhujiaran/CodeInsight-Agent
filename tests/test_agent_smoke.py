from app.agent.agent import CodeAgent
from app.agent.executor import Executor
from app.agent.memory import ConversationMemory
from app.agent.planner import Planner
from app.agent.tool_registry import ToolRegistry
from app.llm.llm import LLMClient
from app.tools.base_tool import BaseTool


class DummySearchTool(BaseTool):
    name = "search_tool"
    description = "dummy search for tests"

    def run(self, input: str) -> str:
        return f"search_result:{input}"


class DummyAnalyzeTool(BaseTool):
    name = "analyze_tool"
    description = "dummy analyze for tests"

    def run(self, input: str) -> str:
        return f"analysis_result:{input}"


def test_code_agent_run_smoke() -> None:
    llm = LLMClient(provider="none", model="dummy-model")
    planner = Planner(llm=llm)

    registry = ToolRegistry()
    registry.register(DummySearchTool())
    registry.register(DummyAnalyzeTool())
    executor = Executor(registry=registry)
    memory = ConversationMemory()

    agent = CodeAgent(planner=planner, executor=executor, llm=llm, memory=memory)
    result = agent.run("请帮我分析登录模块")

    assert len(result.plan) >= 1
    assert len(result.tool_results) >= 1
    assert "Tool Results" in result.context
    assert "search_result" in result.context
    assert isinstance(result.answer, str) and result.answer.strip()

    messages = memory.get_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
