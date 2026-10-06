"""
Human-in-the-loop StateGraph: draft -> approval (interrupt) -> execute | cancel.

The graph pauses at ``approval`` via ``interrupt()``; the proposed action is
returned to the caller. Resume the same ``thread_id`` with
``Command(resume={"approved": bool, "feedback": str})``. Requires a checkpointer.
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from graphs.state import StateFactory, reduce_latest


class ApprovalState(StateFactory, total=False):
    request: str
    proposal: str
    approved: bool
    feedback: str
    result: Annotated[str | None, reduce_latest("result")]


def build_approval_graph(llm: BaseChatModel, checkpointer: BaseCheckpointSaver):
    def draft(state: ApprovalState) -> dict:
        reply = llm.invoke(
            [HumanMessage(content=f"Propose a short action plan for: {state['request']}")]
        )
        return {"proposal": str(reply.content)}

    def approval(state: ApprovalState) -> dict:
        decision = interrupt({"proposal": state["proposal"], "question": "Approve this plan?"})
        return {
            "approved": bool(decision.get("approved")),
            "feedback": decision.get("feedback", ""),
        }

    def route(state: ApprovalState) -> Literal["execute", "cancel"]:
        return "execute" if state.get("approved") else "cancel"

    def execute(state: ApprovalState) -> dict:
        return {"result": f"executed: {state['proposal']}"}

    def cancel(state: ApprovalState) -> dict:
        return {"result": f"cancelled: {state.get('feedback') or 'rejected by reviewer'}"}

    graph = StateGraph(ApprovalState)
    graph.add_node("draft", draft)
    graph.add_node("approval", approval)
    graph.add_node("execute", execute)
    graph.add_node("cancel", cancel)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "approval")
    graph.add_conditional_edges("approval", route, {"execute": "execute", "cancel": "cancel"})
    graph.add_edge("execute", END)
    graph.add_edge("cancel", END)
    return graph.compile(checkpointer=checkpointer)
