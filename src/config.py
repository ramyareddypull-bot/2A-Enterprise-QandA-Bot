"""
LLM & Pinecone Configuration
------------------------------
LLM switching (via .env):
    LLM_PROVIDER=groq    ← testing phase — uses Groq (free tier)
    LLM_PROVIDER=openai  ← production phase — uses OpenAI gpt-4o-mini

Embeddings:
    Always uses HuggingFace sentence-transformers/all-MiniLM-L6-v2.
    Completely free, runs locally, no API key needed.
    Dimension: 384 (Pinecone index must be created with dim=384).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LLM Provider — reads from .env
# ============================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# ============================================================
# Pinecone Settings
# ============================================================
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "enterprise-rag")
PINECONE_DIMENSION  = 384           # all-MiniLM-L6-v2 output dimension (NOT 1536)
PINECONE_NAMESPACES = ["hr", "technical", "compliance"]

# ============================================================
# Retrieval Settings
# ============================================================
TOP_K_PER_NAMESPACE = 4             # chunks retrieved from a single namespace
TOP_K_GENERAL       = 2             # chunks per namespace when routing to "general"
MAX_RETRIES         = 2             # max grader / hallucination retry loops


def get_llm(temperature: float = 0):
    """Return the active LLM based on LLM_PROVIDER env var."""
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )


def get_embeddings():
    """
    Free local embeddings via HuggingFace sentence-transformers.
    Model: all-MiniLM-L6-v2 — 384 dimensions, ~90MB download on first use,
    cached locally after that. No API key required.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_active_provider() -> str:
    """Helper to print which LLM is active."""
    if LLM_PROVIDER == "groq":
        return "Groq (llama-3.3-70b-versatile)"
    return "OpenAI (gpt-4o-mini)"
