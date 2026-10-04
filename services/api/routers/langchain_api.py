"""
/lc endpoints — LangChain facilities (chains, tools, RAG, memory, streaming).

POST /lc/chat          chat with per-session memory (stream=true → SSE)
POST /lc/summarize     summarization chain
POST /lc/structured    structured (Pydantic) extraction chain
POST /lc/tools         one tool-calling round (model picks and runs tools)
POST /lc/rag           retriever chain over the Qdrant knowledge base
DELETE /lc/chat/{id}   forget a session
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel

from chains.lcel import Extraction, structured_chain, summarize_chain
from chains.memory import chat_with_memory, clear_history
from chains.rag import rag_chain
from chains.tools import base_tools
from core.llm_factory import get_chat_model, run_config

router = APIRouter(prefix="/lc", tags=["langchain"])


def get_llm() -> BaseChatModel:
    """Dependency: the configured chat model (default llama3.1-8b)."""
    return get_chat_model()


def get_retriever():
    from config import get_settings
    from core.vector_store import get_vector_store

    return get_vector_store().as_retriever(search_kwargs={"k": get_settings().retrieval_top_k})


class ChatBody(BaseModel):
    message: str
    session_id: str = "default"
    stream: bool = False


@router.post("/chat")
def chat(body: ChatBody, llm: BaseChatModel = Depends(get_llm)):
    chain = chat_with_memory(llm)
    cfg = run_config(configurable={"session_id": body.session_id})
    if not body.stream:
        return {"session_id": body.session_id, "reply": chain.invoke({"input": body.message}, cfg)}

    def events():
        for chunk in chain.stream({"input": body.message}, cfg):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.delete("/chat/{session_id}")
def forget(session_id: str):
    clear_history(session_id)
    return {"cleared": session_id}


class TextBody(BaseModel):
    text: str


@router.post("/summarize")
def summarize(body: TextBody, llm: BaseChatModel = Depends(get_llm)):
    return {"summary": summarize_chain(llm).invoke({"text": body.text}, run_config())}


@router.post("/structured", response_model=Extraction)
def structured(body: TextBody, llm: BaseChatModel = Depends(get_llm)):
    return structured_chain(llm).invoke({"text": body.text}, run_config())


class PromptBody(BaseModel):
    prompt: str


@router.post("/tools")
def tool_call(body: PromptBody, llm: BaseChatModel = Depends(get_llm)):
    tools = {t.name: t for t in base_tools()}
    messages = [HumanMessage(content=body.prompt)]
    ai = llm.bind_tools(list(tools.values())).invoke(messages, run_config())
    messages.append(ai)
    calls = []
    for call in ai.tool_calls:
        try:
            tool = tools.get(call["name"])
            output = tool.invoke(call["args"]) if tool else "unknown tool"
        except Exception as exc:
            output = f"Error: {exc}"
        calls.append({"name": call["name"], "args": call["args"], "output": output})
        messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    final = llm.invoke(messages, run_config()) if calls else ai
    return {"tool_calls": calls, "answer": final.content}


class QuestionBody(BaseModel):
    question: str


@router.post("/rag")
def rag(
    body: QuestionBody, llm: BaseChatModel = Depends(get_llm), retriever=Depends(get_retriever)
):
    return rag_chain(retriever, llm).invoke(body.question, run_config())
