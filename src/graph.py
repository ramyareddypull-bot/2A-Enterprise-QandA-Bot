"""
LangGraph State Graph — RAG Pipeline
--------------------------------------
Wires the 5 nodes together with conditional edges:

  START
    ↓
  [router]       → sets doc_type
    ↓
  [retriever]    → fetches top-K chunks from Pinecone
    ↓
  [grader]       → filters irrelevant chunks
    ↓ (conditional)
  ┌─ if "relevant"     → [generator]
  └─ if "not_relevant" → [retriever]  (retry, broadens to "general")
    ↓
  [generator]    → creates grounded answer
    ↓
  [checker]      → validates no hallucination
    ↓ (conditional)
  ┌─ if "grounded"     → END
  └─ if "not_grounded" → [generator]  (retry, max MAX_RETRIES times)

Usage:
    from src.graph import run_query
    result = run_query("How many sick days do employees get?")
    print(result["generation"])
"""
from langgraph.graph import StateGraph, END

from src.state import RAGState
from src.config import MAX_RETRIES
from src.nodes.router    import route_question
from src.nodes.retriever import retrieve_documents
from src.nodes.grader    import grade_documents
from src.nodes.generator import generate_answer
from src.nodes.checker   import check_hallucination


# ─── Conditional Edge Functions ───────────────────────────────────────────────

def decide_after_grading(state: RAGState) -> str:
    """After grading: generate if we have relevant docs, else retry retrieval."""
    if state["grading_result"] == "relevant":
        return "generate"

    # If already retried too many times, generate anyway (prevents infinite loop)
    if state.get("retry_count", 0) >= MAX_RETRIES:
        print("  ⚠ Grader: no relevant docs after max retries — generating with what we have")
        return "generate"

    print("  ⚠ Grader: no relevant docs — broadening search to 'general'")
    return "retry_retrieve"


def decide_after_checking(state: RAGState) -> str:
    """After hallucination check: finish or ask generator to try again."""
    if state["hallucination_result"] == "grounded":
        return "finish"

    if state.get("retry_count", 0) >= MAX_RETRIES:
        print("  ⚠ Checker: max retries reached — returning current answer")
        return "finish"

    print("  ⚠ Checker: answer not grounded — asking generator to retry")
    return "regenerate"


# ─── Retry helper: widen search scope before re-retrieval ─────────────────────

def widen_and_retrieve(state: RAGState) -> RAGState:
    """Called when grader finds nothing relevant. Switches to 'general' namespace."""
    new_state = {**state, "doc_type": "general", "retry_count": state.get("retry_count", 0) + 1}
    return retrieve_documents(new_state)


# ─── Build the Graph ──────────────────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(RAGState)

    # Register nodes
    workflow.add_node("router",          route_question)
    workflow.add_node("retriever",       retrieve_documents)
    workflow.add_node("grader",          grade_documents)
    workflow.add_node("generator",       generate_answer)
    workflow.add_node("checker",         check_hallucination)
    workflow.add_node("retry_retrieve",  widen_and_retrieve)

    # Entry point
    workflow.set_entry_point("router")

    # Fixed edges
    workflow.add_edge("router",         "retriever")
    workflow.add_edge("retriever",      "grader")
    workflow.add_edge("retry_retrieve", "grader")   # retry flows back into grader
    workflow.add_edge("generator",      "checker")

    # Conditional: after grading
    workflow.add_conditional_edges(
        "grader",
        decide_after_grading,
        {
            "generate":       "generator",
            "retry_retrieve": "retry_retrieve",
        },
    )

    # Conditional: after hallucination check
    workflow.add_conditional_edges(
        "checker",
        decide_after_checking,
        {
            "finish":     END,
            "regenerate": "generator",
        },
    )

    return workflow.compile()


# ─── Singleton (imported by app.py and evaluate.py) ──────────────────────────

_graph = None

def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(question: str) -> dict:
    """
    Run a question through the full RAG pipeline.

    Returns the final RAGState dict with keys:
      question, doc_type, documents, generation,
      grading_result, hallucination_result, retry_count
    """
    initial_state: RAGState = {
        "question":            question,
        "doc_type":            "",
        "documents":           [],
        "generation":          "",
        "grading_result":      "",
        "hallucination_result": "",
        "retry_count":         0,
    }
    return _get_graph().invoke(initial_state)
