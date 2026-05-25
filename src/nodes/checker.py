"""
Node 5 — Hallucination Checker
--------------------------------
Verifies that the generated answer is actually grounded in the retrieved documents.
Asks the LLM: "Is this answer supported by the documents? Yes or No."

If "yes" (grounded)     → graph terminates → answer is returned to user
If "no"  (not_grounded) → graph loops back to generator to regenerate
                          (capped at MAX_RETRIES to prevent infinite loops)

Cost: 1 LLM call per check.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.config import get_llm, MAX_RETRIES
from src.state import RAGState

CHECKER_SYSTEM_PROMPT = """You are a hallucination detector. Determine whether an AI-generated answer is fully supported by the provided source documents.

Score "yes"  → The answer contains ONLY information that appears in the documents.
Score "no"   → The answer contains facts, numbers, or claims NOT found in the documents.

Respond with JSON only. Format: {{"score": "yes"}} or {{"score": "no"}}
No explanation, no extra text."""

checker_prompt = ChatPromptTemplate.from_messages([
    ("system", CHECKER_SYSTEM_PROMPT),
    ("human", "Source Documents:\n{documents}\n\nGenerated Answer:\n{generation}"),
])


def check_hallucination(state: RAGState) -> RAGState:
    """Check whether the generated answer is grounded in the retrieved documents."""
    print("\n🛡️  [Checker] Verifying answer is grounded...")

    retry_count = state.get("retry_count", 0)

    # If no documents were retrieved, we can't check — pass through
    if not state["documents"]:
        print("  → No documents to check against — marking as grounded")
        return {**state, "hallucination_result": "grounded", "retry_count": retry_count + 1}

    # If we've already hit max retries, accept the answer regardless
    if retry_count >= MAX_RETRIES:
        print(f"  → Max retries ({MAX_RETRIES}) reached — accepting answer as-is")
        return {**state, "hallucination_result": "grounded", "retry_count": retry_count + 1}

    llm    = get_llm(temperature=0)
    parser = JsonOutputParser()
    chain  = checker_prompt | llm | parser

    # Truncate context to avoid token limits
    context = "\n\n".join([doc.page_content[:800] for doc in state["documents"]])

    try:
        result = chain.invoke({
            "documents":  context,
            "generation": state["generation"],
        })
        score               = result.get("score", "yes").strip().lower()
        hallucination_result = "grounded" if score == "yes" else "not_grounded"
    except Exception as e:
        print(f"  WARNING: Checker parse error ({e}) — defaulting to grounded")
        hallucination_result = "grounded"

    print(f"  → Hallucination check result: '{hallucination_result}'")
    return {**state, "hallucination_result": hallucination_result, "retry_count": retry_count + 1}
