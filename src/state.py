"""
RAG Pipeline State Definition
------------------------------
This TypedDict flows through every LangGraph node.
Every node receives the full state and returns an updated copy.
"""
from typing import TypedDict, List
from langchain_core.documents import Document


class RAGState(TypedDict):
    question: str                # The user's original question
    doc_type: str                # Routed namespace: "hr" | "technical" | "compliance" | "general"
    documents: List[Document]    # Retrieved (and graded) chunks from Pinecone
    generation: str              # LLM-generated answer
    grading_result: str          # "relevant" | "not_relevant"
    hallucination_result: str    # "grounded" | "not_grounded"
    retry_count: int             # Tracks retries to prevent infinite loops
