"""
Node 4 — Answer Generator
---------------------------
Generates a grounded answer from the filtered, relevant chunks.

Rules enforced in the prompt:
  - Answer ONLY from the provided context
  - If context is insufficient → say so explicitly (no hallucination)
  - Do NOT mention filenames in the answer (shown separately in UI)
  - Professional, concise tone

Cost: 1 LLM call per generation attempt.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import get_llm
from src.state import RAGState

GENERATOR_SYSTEM_PROMPT = """You are an enterprise knowledge assistant for ACME Corporation.
Answer the user's question using ONLY the information in the provided context documents.

Rules:
1. Base your answer strictly on the context. Do not use prior knowledge.
2. If the context does not contain enough information to answer the question, say:
   "I couldn't find a definitive answer in the knowledge base. The question may be out of scope or not covered in the available documents."
3. Do NOT mention any filenames or document names in your answer. Source references are shown separately in the UI.
4. Be concise, professional, and factual.
5. Do NOT make up numbers, dates, or policies not stated in the context.

Context:
{context}"""

generator_prompt = ChatPromptTemplate.from_messages([
    ("system", GENERATOR_SYSTEM_PROMPT),
    ("human", "Question: {question}"),
])


def _format_context(documents) -> str:
    """Format retrieved chunks into a readable context string."""
    if not documents:
        return "No relevant documents were retrieved."

    parts = []
    for doc in documents:
        parts.append(doc.page_content)
    return "\n\n---\n\n".join(parts)


def generate_answer(state: RAGState) -> RAGState:
    """Generate an answer grounded in retrieved documents."""
    print("\n✨ [Generator] Generating answer...")

    llm   = get_llm(temperature=0)
    chain = generator_prompt | llm | StrOutputParser()

    context    = _format_context(state["documents"])
    generation = chain.invoke({
        "context":  context,
        "question": state["question"],
    })

    print(f"  → Generated {len(generation)} characters")
    return {**state, "generation": generation}
