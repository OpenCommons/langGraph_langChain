"""Prompt templates and LCEL chains: summarization, Q&A, structured output."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a precise summarizer. Write at most {max_sentences} sentences."),
        ("human", "Summarize the following text:\n\n{text}"),
    ]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the context. "
            "If the context is insufficient, say you do not know.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


class Extraction(BaseModel):
    """Structured result of ``structured_chain``."""

    title: str = Field(description="A short title for the text")
    summary: str = Field(description="One-sentence summary")
    keywords: list[str] = Field(description="Up to five keywords")


def summarize_chain(llm: BaseChatModel) -> Runnable:
    return SUMMARIZE_PROMPT.partial(max_sentences=3) | llm | StrOutputParser()


def qa_chain(llm: BaseChatModel) -> Runnable:
    return QA_PROMPT | llm | StrOutputParser()


def structured_chain(llm: BaseChatModel, schema: type[BaseModel] = Extraction) -> Runnable:
    """prompt | llm | PydanticOutputParser — works with any chat model (no native JSON mode)."""
    parser = PydanticOutputParser(pydantic_object=schema)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Extract structured data. Reply with JSON only.\n{format_instructions}"),
            ("human", "{text}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser
