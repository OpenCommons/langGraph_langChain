"""
/lg endpoints — LangGraph facilities (checkpointed agent, HITL, supervisor).

POST /lg/agent/run             ReAct agent, durable per thread_id
POST /lg/agent/stream          same, streaming graph events as SSE
GET  /lg/agent/state/{thread}  checkpointed messages for a thread
POST /lg/approval/start        draft a plan and pause for human approval
POST /lg/approval/resume       resume a paused thread with a decision
POST /lg/supervisor/run        supervisor → workers multi-agent workflow
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from chains.tools import base_tools
from config import get_settings
from core.llm_factory import get_chat_model, run_config
from governance import get_engine
from governance.tools import govern_tools
from graphs.approval_graph import build_approval_graph
from graphs.checkpoint import get_checkpointer
from graphs.react_agent import build_react_agent
from graphs.supervisor_graph import build_supervisor_graph, initial_state

router = APIRouter(prefix="/lg", tags=["langgraph"])


def get_llm() -> BaseChatModel:
    return get_chat_model()


def jsonable(value: Any) -> Any:
    """Make graph updates JSON-serializable (messages become small dicts)."""
    if isinstance(value, BaseMessage):
        out = {"type": value.type, "content": value.content}
        if getattr(value, "tool_calls", None):
            out["tool_calls"] = value.tool_calls
        return out
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, str | int | float | bool | None):
        return value
    return str(value)


def _config(thread_id: str) -> dict:
    return run_config(thread_id, recursion_limit=get_settings().graph_recursion_limit)


class AgentBody(BaseModel):
    message: str
    thread_id: str | None = None
    # Sender's Lamport timestamp, if the caller keeps a logical clock.
    lamport: int | None = None


def _thread_texts(thread_id: str) -> list[str]:
    """Text of every message already checkpointed for the thread."""
    tup = get_checkpointer().get_tuple(_config(thread_id))
    messages = tup.checkpoint["channel_values"].get("messages", []) if tup else []
    return [str(m.content) for m in messages]


def _agent_tools(body: AgentBody, thread_id: str, token: str | None):
    """Base tools, each call verified over the whole thread and recorded (governance/)."""
    s = get_settings()
    if not s.governance_enabled:
        return base_tools(), None
    engine = get_engine()
    if s.governance_require_token and not token:
        raise HTTPException(401, "X-Capability-Token required")
    lamport = engine.clock.receive(body.lamport) if body.lamport else engine.clock.tick()
    context = [*_thread_texts(thread_id), body.message]
    return govern_tools(
        base_tools(), engine, thread_id=thread_id, context=context, token=token
    ), lamport


@router.post("/agent/run")
def agent_run(
    body: AgentBody,
    response: Response,
    llm: BaseChatModel = Depends(get_llm),
    x_capability_token: str | None = Header(default=None),
):
    thread_id = body.thread_id or str(uuid.uuid4())
    tools, lamport = _agent_tools(body, thread_id, x_capability_token)
    graph = build_react_agent(llm, tools, get_checkpointer())
    out = graph.invoke({"messages": [HumanMessage(content=body.message)]}, _config(thread_id))
    if lamport is not None:
        # Lamport time after the whole run (every tool admission ticked the clock).
        response.headers["X-Lamport"] = str(get_engine().clock.value)
    return {"thread_id": thread_id, "answer": out["messages"][-1].content}


@router.post("/agent/stream")
def agent_stream(
    body: AgentBody,
    llm: BaseChatModel = Depends(get_llm),
    x_capability_token: str | None = Header(default=None),
):
    thread_id = body.thread_id or str(uuid.uuid4())
    tools, lamport = _agent_tools(body, thread_id, x_capability_token)
    graph = build_react_agent(llm, tools, get_checkpointer())

    def events():
        head = (
            {"thread_id": thread_id}
            if lamport is None
            else {"thread_id": thread_id, "lamport": lamport}
        )
        yield f"data: {json.dumps(head)}\n\n"
        for update in graph.stream(
            {"messages": [HumanMessage(content=body.message)]},
            _config(thread_id),
            stream_mode="updates",
        ):
            yield f"data: {json.dumps(jsonable(update))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/agent/state/{thread_id}")
def agent_state(thread_id: str, llm: BaseChatModel = Depends(get_llm)):
    graph = build_react_agent(llm, base_tools(), get_checkpointer())
    values = graph.get_state(_config(thread_id)).values
    return {"thread_id": thread_id, "messages": jsonable(values.get("messages", []))}


class ApprovalStart(BaseModel):
    request: str
    thread_id: str | None = None


class ApprovalResume(BaseModel):
    thread_id: str
    approved: bool
    feedback: str = ""


def _approval_result(graph, thread_id: str, out: dict) -> dict:
    pending = graph.get_state(_config(thread_id)).tasks
    interrupts = [i.value for t in pending for i in t.interrupts]
    return {
        "thread_id": thread_id,
        "status": "awaiting_approval" if interrupts else "done",
        "interrupts": jsonable(interrupts),
        "result": out.get("result"),
    }


@router.post("/approval/start")
def approval_start(body: ApprovalStart, llm: BaseChatModel = Depends(get_llm)):
    thread_id = body.thread_id or str(uuid.uuid4())
    graph = build_approval_graph(llm, get_checkpointer())
    out = graph.invoke({"request": body.request}, _config(thread_id))
    return _approval_result(graph, thread_id, out)


@router.post("/approval/resume")
def approval_resume(body: ApprovalResume, llm: BaseChatModel = Depends(get_llm)):
    graph = build_approval_graph(llm, get_checkpointer())
    decision = {"approved": body.approved, "feedback": body.feedback}
    out = graph.invoke(Command(resume=decision), _config(body.thread_id))
    return _approval_result(graph, body.thread_id, out)


class SupervisorBody(BaseModel):
    task: str
    thread_id: str | None = None


@router.post("/supervisor/run")
def supervisor_run(body: SupervisorBody, llm: BaseChatModel = Depends(get_llm)):
    thread_id = body.thread_id or str(uuid.uuid4())
    graph = build_supervisor_graph(llm, get_checkpointer())
    out = graph.invoke(initial_state(body.task), _config(thread_id))
    return {"thread_id": thread_id, "messages": jsonable(out["messages"]), "steps": out["steps"]}
