"""Conversation memory: per-session chat history wired into an LCEL chain."""

from __future__ import annotations

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

MAX_SESSIONS = 1000
_histories: dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _histories:
        if len(_histories) >= MAX_SESSIONS:
            _histories.pop(next(iter(_histories)))  # evict the oldest session
        _histories[session_id] = InMemoryChatMessageHistory()
    return _histories[session_id]


def clear_history(session_id: str) -> None:
    _histories.pop(session_id, None)


def chat_with_memory(llm: BaseChatModel, system_prompt: str | None = None) -> Runnable:
    """Chain taking ``{"input": str}`` and a ``session_id`` in ``configurable``."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt or "You are a helpful assistant."),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    return RunnableWithMessageHistory(
        prompt | llm | StrOutputParser(),
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )
