"""
Node 1 — Query Router
-----------------------
Takes the user's question and classifies it into one of four namespaces:
  hr | technical | compliance | general

Uses an LLM call (Groq or OpenAI) to do the classification.
Cost: 1 LLM call per query.
"""
from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm
from src.state import RAGState

ROUTER_SYSTEM_PROMPT = """You are an expert at routing user questions to the correct enterprise document category.

Classify the question into exactly ONE of these categories:
- "hr"         : leave policy, benefits, performance reviews, remote work, onboarding, code of conduct, salary, disciplinary
- "technical"  : APIs, authentication, rate limits, error codes, deployments, system architecture, SDKs, webhooks
- "compliance" : GDPR, data privacy, data retention, incident response, vendor management, audits, security policy, acceptable use
- "general"    : the question spans multiple categories, is unclear, or doesn't fit any of the above

IMPORTANT: Respond with ONLY the single lowercase category name. No explanation, no punctuation."""

router_prompt = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM_PROMPT),
    ("human", "Question: {question}"),
])

VALID_DOC_TYPES = {"hr", "technical", "compliance", "general"}


def route_question(state: RAGState) -> RAGState:
    """Classify the question and set doc_type in state."""
    print("\n🔀 [Router] Classifying question...")

    llm   = get_llm(temperature=0)
    chain = router_prompt | llm

    result   = chain.invoke({"question": state["question"]})
    doc_type = result.content.strip().lower().strip('"').strip("'")

    # Sanitise: if LLM returns something unexpected, fall back to general
    if doc_type not in VALID_DOC_TYPES:
        print(f"  WARNING: LLM returned unexpected type '{doc_type}' — using 'general'")
        doc_type = "general"

    print(f"  → Routed to namespace: '{doc_type}'")
    return {**state, "doc_type": doc_type}
