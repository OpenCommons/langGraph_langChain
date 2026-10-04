"""RAG: document loading and splitting, ingestion, retriever and retriever chain."""

from __future__ import annotations

import pathlib

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chains.lcel import QA_PROMPT


def load_documents(path: str | pathlib.Path) -> list[Document]:
    """Load a .pdf, or a text-like file (.txt/.md/...), as LangChain documents."""
    p = pathlib.Path(path)
    if p.suffix.lower() == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(p)).load()
    return [Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})]


def split_documents(
    docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 100
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def ingest(store: VectorStore, docs: list[Document]) -> int:
    """Split and add documents to a vector store; returns the number of chunks."""
    chunks = split_documents(docs)
    if chunks:
        store.add_documents(chunks)
    return len(chunks)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata.get('source', 'unknown')}] {d.page_content}" for d in docs)


def rag_chain(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable:
    """question -> {answer, sources}: retrieve, stuff into the Q&A prompt, generate."""

    def _retrieve(question: str) -> dict:
        docs = retriever.invoke(question)
        return {"question": question, "docs": docs, "context": format_docs(docs)}

    answer = QA_PROMPT | llm | StrOutputParser()
    return (
        RunnableLambda(_retrieve)
        | RunnablePassthrough.assign(answer=answer)
        | RunnableLambda(
            lambda r: {
                "answer": r["answer"],
                "sources": sorted({d.metadata.get("source", "unknown") for d in r["docs"]}),
            }
        )
    )
