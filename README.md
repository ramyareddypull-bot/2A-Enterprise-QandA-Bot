# Enterprise Policy Q&A Bot
**Project 2A — Gen Academy | Intermediate RAG Pipeline**

A production-grade Retrieval-Augmented Generation (RAG) system built on **LangGraph + Pinecone + Groq/OpenAI**. It answers natural-language questions over enterprise documents (HR policies, technical docs, compliance manuals) using a stateful 5-node pipeline with relevance grading and hallucination checking.

---

## Architecture

```
User Question
     │
     ▼
  [Router]          ← classifies question → namespace (hr / technical / compliance / general)
     │
     ▼
 [Retriever]        ← semantic search in Pinecone (HuggingFace embeddings, free)
     │
     ▼
  [Grader]          ← filters irrelevant chunks before generation
     │
     ├── not relevant → retry with broader search
     │
     ▼
 [Generator]        ← LLM generates grounded answer from filtered chunks
     │
     ▼
  [Checker]         ← verifies answer is grounded in retrieved docs
     │
     ├── not grounded → regenerate (max 2 retries)
     │
     ▼
   Answer
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph) |
| LLM — Testing | Groq `llama-3.1-70b-versatile` |
| LLM — Production | OpenAI `gpt-4o-mini` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (free, local) |
| Vector DB | Pinecone (serverless, namespaced) |
| UI | Streamlit |

---

## Project Structure

```
enterprise-rag-bot/
│
├── src/
│   ├── state.py          ← RAGState TypedDict (shared across all nodes)
│   ├── config.py         ← LLM/embedding/Pinecone config + provider switch
│   ├── ingestion.py      ← One-time doc loading → chunking → Pinecone upsert
│   ├── graph.py          ← LangGraph StateGraph: wires all 5 nodes + edges
│   └── nodes/
│       ├── router.py     ← Node 1: classifies question into namespace
│       ├── retriever.py  ← Node 2: Pinecone similarity search
│       ├── grader.py     ← Node 3: per-chunk relevance filtering
│       ├── generator.py  ← Node 4: grounded answer generation
│       └── checker.py    ← Node 5: hallucination detection
│
├── docs/
│   ├── hr/               ← HR policy documents
│   ├── technical/        ← Technical documentation
│   └── compliance/       ← Compliance & security manuals
│
├── app.py                ← Streamlit chat UI
├── evaluate.py           ← 15-question evaluation script → evaluation_report.json
├── requirements.txt
├── .env.example          ← API key template (safe to commit)
└── README.md
```

---

## Files to Commit to Git

Push these files — **do not push** `.env`, `venv/`, `tests/`, or `evaluation_report.json`.

```
✅ src/__init__.py
✅ src/state.py
✅ src/config.py
✅ src/ingestion.py
✅ src/graph.py
✅ src/nodes/__init__.py
✅ src/nodes/router.py
✅ src/nodes/retriever.py
✅ src/nodes/grader.py
✅ src/nodes/generator.py
✅ src/nodes/checker.py
✅ app.py
✅ evaluate.py
✅ requirements.txt
✅ .env.example
✅ docs/hr/hr_policy.txt
✅ docs/technical/tech_docs.txt
✅ docs/compliance/compliance_manual.txt
✅ README.md

❌ .env                   ← real API keys — never commit
❌ venv/                  ← virtual environment
❌ tests/                 ← node-by-node unit tests (local dev only)
❌ evaluation_report.json ← generated output
❌ __pycache__/
```

**Recommended `.gitignore`:**
```
.env
venv/
__pycache__/
*.pyc
evaluation_report.json
.pytest_cache/
```

---

## Setup

```powershell
# 1. Create and activate virtual environment (Windows)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
copy .env.example .env
# Edit .env — add your Groq, OpenAI, and Pinecone API keys

# 4. Ingest documents into Pinecone (run once)
python -m src.ingestion

# 5. Launch the app
streamlit run app.py

# 6. Run the 15-question evaluation
python evaluate.py
```

---

## Switching LLM Provider

Change one line in `.env`:

```
LLM_PROVIDER=groq    # testing — Groq llama-3.1-70b (free tier)
LLM_PROVIDER=openai  # production — OpenAI gpt-4o-mini
```

Embeddings always use HuggingFace `all-MiniLM-L6-v2` regardless of LLM provider.

---

## Evaluation

`evaluate.py` runs 15 curated questions through the full pipeline:

| Category | Count | Purpose |
|---|---|---|
| HR | 4 | Direct factual retrieval |
| Technical | 3 | API / error code lookup |
| Compliance | 3 | GDPR / incident response |
| Edge Cases | 5 | Cross-doc, ambiguous, out-of-scope |

Output: `evaluation_report.json` with per-question routing accuracy, retrieval count, grading result, hallucination check, and final answer.
