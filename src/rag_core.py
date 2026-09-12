"""
Company Policy RAG Agent — shared retrieval-augmented generation (RAG) logic.

This module holds the retrieval, prompting, and citation logic used by BOTH
notebooks/policy_rag_agent.ipynb (for interactive testing) and src/agent.py
(the model that gets registered to Unity Catalog and served). It's generated
programmatically by the notebook; this copy is kept here for readability and
version control.
"""

import os
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


SYSTEM_PROMPT = (
    "You are a helpful company policy assistant. Answer the user's question using "
    "ONLY the provided context. Every factual claim you make must be immediately "
    "followed by a citation in the form (Source: <file>, page <n>). "
    "If the context does not contain the answer, say you do not know -- do not guess."
)


def format_docs(docs):
    """Turn a list of retrieved chunks into the context string the LLM sees,
    with a citation tag (source file, page, section) in front of each chunk."""
    blocks = []
    for d in docs:
        src = os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", "?")
        section = d.metadata.get("section") or "General"
        blocks.append(f"[Source: {src}, page {page}, section: {section}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def get_sources(docs):
    """Build a structured, de-duplicated citation list (source file, page, section)
    from the retrieved chunks, so the caller can display exactly what was used
    to answer the question without relying on the model to cite itself correctly."""
    seen = set()
    sources = []
    for d in docs:
        src = os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page")
        section = d.metadata.get("section")
        key = (src, page, section)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": src, "page": page, "section": section})
    return sources


def make_hybrid_retriever(vector_retriever, bm25_retriever):
    """Combine keyword (BM25) and vector search results for a query.

    Both retrievers are queried, results are interleaved, and duplicate chunks
    (identical text content) are dropped. This is done with a small manual
    merge rather than LangChain's built-in EnsembleRetriever class, to keep
    this module independent of that class's exact import path across
    LangChain versions."""
    def hybrid_retrieve(query):
        vector_docs = vector_retriever.invoke(query)
        keyword_docs = bm25_retriever.invoke(query)
        seen = set()
        merged = []
        for pair in zip(keyword_docs, vector_docs):
            for doc in pair:
                if doc.page_content in seen:
                    continue
                seen.add(doc.page_content)
                merged.append(doc)
        return merged
    return hybrid_retrieve


def make_build_prompt(tokenizer):
    """Return a function that turns {context, question} into a fully formatted
    prompt string, using the given tokenizer's chat template."""
    def build_prompt(inputs):
        user_content = f"Context:\n{inputs['context']}\n\nQuestion: {inputs['question']}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return build_prompt


def build_rag_chain(vector_retriever, bm25_retriever, tokenizer, llm):
    """Assemble the full RAG chain: given a question, retrieve chunks once
    using hybrid search, then build both the generated answer and the
    structured citation list from that same set of retrieved chunks."""
    hybrid_retrieve = make_hybrid_retriever(vector_retriever, bm25_retriever)
    build_prompt = make_build_prompt(tokenizer)

    def retrieve_and_package(query):
        docs = hybrid_retrieve(query)
        return {
            "context": format_docs(docs),
            "sources": get_sources(docs),
            "question": query,
        }

    retrieval_step = RunnableLambda(retrieve_and_package)

    rag_chain = (
        retrieval_step
        | RunnableParallel(
            answer=RunnableLambda(build_prompt) | llm | StrOutputParser(),
            sources=RunnableLambda(lambda x: x["sources"]),
        )
    )
    return rag_chain
