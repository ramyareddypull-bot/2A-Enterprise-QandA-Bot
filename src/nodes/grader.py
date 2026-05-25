"""
Node 3 — Relevance Grader
---------------------------
For each retrieved chunk, asks the LLM: "Is this document relevant to the question?"
Filters out irrelevant chunks before they pollute the answer.

If ≥ 1 chunk passes → grading_result = "relevant"  → proceed to generator
If 0 chunks pass   → grading_result = "not_relevant" → graph retries retriever

Cost: 1 LLM call per chunk (batched sequentially). To reduce Groq calls during
testing you can lower TOP_K_PER_NAMESPACE in config.py.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.config import get_llm
from src.state import RAGState

GRADER_SYSTEM_PROMPT = """You are a relevance grader. Assess whether a retrieved document is relevant to a user question.

A document is relevant if it contains keywords, facts, or semantic meaning that could help answer the question.
Be generous — partial relevance counts.

Respond with JSON only. Format: {{"score": "yes"}} or {{"score": "no"}}
No explanation, no extra text."""

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", GRADER_SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nDocument:\n{document}"),
])


def grade_documents(state: RAGState) -> RAGState:
    """Filter documents to only those relevant to the question."""
    print("\n✅ [Grader] Evaluating document relevance...")

    llm    = get_llm(temperature=0)
    parser = JsonOutputParser()
    chain  = grader_prompt | llm | parser

    question      = state["question"]
    documents     = state["documents"]
    relevant_docs = []

    for i, doc in enumerate(documents):
        try:
            result = chain.invoke({
                "question": question,
                "document": doc.page_content[:1500],   # truncate to stay within limits
            })
            score = result.get("score", "no").strip().lower()
            if score == "yes":
                relevant_docs.append(doc)
                print(f"  Doc {i+1}: ✓ relevant  [{doc.metadata.get('source','?')}]")
            else:
                print(f"  Doc {i+1}: ✗ filtered  [{doc.metadata.get('source','?')}]")
        except Exception as e:
            # On parse failure, keep the document (fail open)
            print(f"  Doc {i+1}: parse error ({e}) — keeping doc")
            relevant_docs.append(doc)

    grading_result = "relevant" if relevant_docs else "not_relevant"
    print(f"  → {len(relevant_docs)}/{len(documents)} chunks kept → '{grading_result}'")

    return {**state, "documents": relevant_docs, "grading_result": grading_result}
