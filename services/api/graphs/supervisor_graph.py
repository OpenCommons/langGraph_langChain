"""
Multi-agent supervisor workflow.

A supervisor LLM routes the task to a worker (``math`` — a ReAct agent with the
calculator, ``writer`` — a plain LLM) or ``FINISH``; control returns to the
supervisor after each worker, bounded by ``max_steps``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from chains.tools import calculator
from graphs.react_agent import build_react_agent
from graphs.state import StateFactory, reduce_max

WORKERS = ("math", "writer")
Route = Literal["math", "writer", "FINISH"]

SUPERVISOR_PROMPT = (
    "You are a supervisor routing work to workers: math (arithmetic via a calculator) "
    "and writer (prose). Reply with exactly one word: math, writer, or FINISH when "
    "the task is complete."
)


class SupervisorState(StateFactory, total=False):
    next: str
    steps: Annotated[int, reduce_max("steps")]


def parse_route(text: str) -> Route:
    """Map free-form supervisor output to a route; anything unrecognized finishes."""
    lowered = text.strip().lower()
    for worker in WORKERS:
        if worker in lowered:
            return worker  # type: ignore[return-value]
    return "FINISH"


def build_supervisor_graph(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    max_steps: int = 4,
):
    math_agent = build_react_agent(llm, [calculator])

    def supervisor(state: SupervisorState) -> dict:
        steps = state.get("steps", 0)
        if steps >= max_steps:
            return {"next": "FINISH"}
        reply = llm.invoke([SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"])
        return {"next": parse_route(str(reply.content)), "steps": steps + 1}

    def math(state: SupervisorState) -> dict:
        out = math_agent.invoke({"messages": state["messages"]})
        return {"messages": [AIMessage(content=f"[math] {out['messages'][-1].content}")]}

    def writer(state: SupervisorState) -> dict:
        reply = llm.invoke([SystemMessage(content="You are a concise writer.")] + state["messages"])
        return {"messages": [AIMessage(content=f"[writer] {reply.content}")]}

    def route(state: SupervisorState) -> str:
        next_node = state.get("next", "FINISH")
        return next_node if next_node in WORKERS else END

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("math", math)
    graph.add_node("writer", writer)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route, {"math": "math", "writer": "writer", END: END})
    graph.add_edge("math", "supervisor")
    graph.add_edge("writer", "supervisor")
    return graph.compile(checkpointer=checkpointer)


def initial_state(task: str) -> SupervisorState:
    return {"messages": [HumanMessage(content=task)], "steps": 0}
