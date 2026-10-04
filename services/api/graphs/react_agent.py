"""
Reusable ReAct-style StateGraph with tool calling, conditional routing and
checkpointing.

    START -> agent --(tool calls?)--> tools -> agent ... -> END

Pass a checkpointer and a ``thread_id`` (in ``configurable``) to make a
conversation durable across calls.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools when they help, then answer concisely."
)


def route_after_agent(state: MessagesState) -> Literal["tools", "end"]:
    """Conditional edge: run tools if the model requested any, otherwise finish."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


def build_react_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    bound = llm.bind_tools(tools)

    def agent(state: MessagesState) -> dict:
        return {
            "messages": [bound.invoke([SystemMessage(content=system_prompt)] + state["messages"])]
        }

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
