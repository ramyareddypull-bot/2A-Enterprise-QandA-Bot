"""
Node 2 — Pinecone Retriever
-----------------------------
Fetches the top-K most semantically similar chunks from Pinecone.

- If doc_type is "hr" / "technical" / "compliance" → search that single namespace
- If doc_type is "general" → search ALL three namespaces and merge results

No LLM call here. Cost: only embedding inference (OpenAI).
"""
from langchain_pinecone import PineconeVectorStore
from src.config import (
    get_embeddings,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACES,
    TOP_K_PER_NAMESPACE,
    TOP_K_GENERAL,
)
from src.state import RAGState


def _get_vectorstore(namespace: str) -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
        namespace=namespace,
    )


def retrieve_documents(state: RAGState) -> RAGState:
    """Search Pinecone and return top-K chunks."""
    print("\n🔍 [Retriever] Searching Pinecone...")

    doc_type = state["doc_type"]
    question = state["question"]
    all_docs  = []

    if doc_type == "general":
        # Search all namespaces and take TOP_K_GENERAL from each
        for ns in PINECONE_NAMESPACES:
            vs   = _get_vectorstore(ns)
            docs = vs.similarity_search(question, k=TOP_K_GENERAL)
            for doc in docs:
                doc.metadata.setdefault("doc_type", ns)
            all_docs.extend(docs)
        print(f"  → Searched all namespaces, retrieved {len(all_docs)} total chunks")
    else:
        vs       = _get_vectorstore(doc_type)
        all_docs = vs.similarity_search(question, k=TOP_K_PER_NAMESPACE)
        print(f"  → Retrieved {len(all_docs)} chunks from namespace '{doc_type}'")

    return {**state, "documents": all_docs}
