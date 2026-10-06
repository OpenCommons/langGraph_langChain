"""Wrap LangChain tools so every call is admitted through the Atomic Triad."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.tools import BaseTool, StructuredTool

from governance.engine import GovernanceEngine


def blocked_message(reason: str | None) -> str:
    return f"BLOCKED by governance ({reason})."


def govern_tools(
    tools: Sequence[BaseTool],
    engine: GovernanceEngine,
    *,
    thread_id: str,
    context: Sequence[str],
    token: str | None = None,
) -> list[BaseTool]:
    """Return copies of ``tools`` whose execution is verified and recorded.

    ``context`` is the full thread so far (prior messages + the new prompt); the
    proposed call's arguments are added to it before verification. A refused
    call returns a BLOCKED message to the model instead of running.
    """

    def wrap(tool: BaseTool) -> BaseTool:
        def run(**kwargs):
            admission = engine.admit(
                thread_id=thread_id,
                tool=tool.name,
                args=kwargs,
                context=context,
                execute=lambda: tool.invoke(kwargs),
                token=token,
            )
            return admission.result if admission.allowed else blocked_message(admission.reason)

        return StructuredTool(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            func=run,
        )

    return [wrap(t) for t in tools]
