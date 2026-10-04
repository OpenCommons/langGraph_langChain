"""Fake chat model for tests: scripted replies, no live LLM, supports bind_tools."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """Returns ``replies`` in order (the last one repeats once exhausted)."""

    replies: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


def tool_call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def fake_llm(*texts: str | AIMessage) -> ScriptedChatModel:
    return ScriptedChatModel(
        replies=[AIMessage(content=t) if isinstance(t, str) else t for t in texts]
    )
