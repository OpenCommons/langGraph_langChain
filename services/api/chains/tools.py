"""Tool definitions shared by the LangChain tool-calling demo and LangGraph agents."""

from __future__ import annotations

import ast
import operator
from datetime import UTC, datetime

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPONENT = 1000
_MAX_BASE = 1_000_000


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > _MAX_EXPONENT or abs(left) > _MAX_BASE):
            raise ValueError("power too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def safe_calculate(expression: str) -> float:
    """Evaluate +, -, *, /, //, %, ** over numbers only (no names, calls or attributes)."""
    return _eval(ast.parse(expression.strip(), mode="eval").body)


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(12 + 3) * 4 / 5'."""
    try:
        return str(safe_calculate(expression))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
        return f"Error: {exc}"


@tool
def current_time() -> str:
    """Return the current UTC date and time in ISO-8601 format."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def make_retrieval_tool(retriever: BaseRetriever, k_chars: int = 1500) -> BaseTool:
    """Wrap any retriever as a tool the model can call."""

    @tool
    def retrieve_documents(query: str) -> str:
        """Search the knowledge base for passages relevant to the query."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found."
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content[:k_chars]}"
            for d in docs
        )

    return retrieve_documents


def base_tools() -> list[BaseTool]:
    return [calculator, current_time]
