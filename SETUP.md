# Enterprise RAG Bot — Complete Setup Guide (Windows + VSCode)

---

## Prerequisites

- Python 3.10 or higher
- VSCode installed
- Groq API key → https://console.groq.com (used as LLM during testing)
- OpenAI API key → https://platform.openai.com/api-keys (used as LLM in production only — NOT for embeddings)
- Pinecone API key → https://app.pinecone.io (free Starter plan is fine)

> **Embeddings are free.** This project uses `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace.
> It runs locally on your CPU, downloads ~90MB on first use, and costs nothing.

---

## Step 1 — Open the Project in VSCode

```
Open VSCode → File → Open Folder → select the `enterprise-rag-bot` folder
```

Open the integrated terminal: **Ctrl + `** (backtick)

---

## Step 2 — Create and Activate Virtual Environment

```powershell
# Create venv
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Confirm activation — you should see (venv) in your prompt
```

> VSCode will auto-detect the venv. If prompted "New virtual environment detected — use it?", click **Yes**.
> If not, press `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `./venv/Scripts/python.exe`

---

## Step 3 — Install Dependencies

```powershell
pip install -r requirements.txt
```

This installs: LangChain, LangGraph, LangChain-Groq, LangChain-OpenAI, LangChain-Pinecone,
LangChain-HuggingFace, sentence-transformers, Pinecone client, Streamlit, PyPDF, python-dotenv, and pytest.

> The `sentence-transformers` package (~90MB) downloads the embedding model automatically
> the first time you run ingestion. After that it's cached locally — no repeated downloads.

---

## Step 4 — Set Up Environment Variables

```powershell
# Copy the example file
copy .env.example .env
```

Open `.env` in VSCode and fill in your real API keys:

```
LLM_PROVIDER=groq                          ← keep this as "groq" for now

GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx      ← LLM for testing
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx     ← LLM for production only (NOT used for embeddings)
PINECONE_API_KEY=pcsk_xxxxxxxxxxxxxxxxxxxx

PINECONE_INDEX_NAME=enterprise-rag
```

> **Note:** The `.env` file is gitignored. Never commit real API keys.
> **Embeddings:** No API key needed — HuggingFace runs locally for free.

---

## Step 5 — Ingest Documents into Pinecone

This creates the Pinecone index and uploads your documents. Run once.

```powershell
python -m src.ingestion
```

Expected output:
```
===================================================
ENTERPRISE RAG BOT — DOCUMENT INGESTION
===================================================

[Step 1] Setting up Pinecone index...
  Creating Pinecone index 'enterprise-rag'...
  Index created and ready.

[Step 2/2] Processing namespace: HR
  Loaded: hr_policy.txt  (1 page(s))
  Chunked into 22 pieces (chunk_size=512, overlap=64)
  ✓ Done — 22 chunks indexed in 'hr'

[Step 3/3] Processing namespace: TECHNICAL
  ...

✅  Ingestion complete! Your knowledge base is ready.
```

> This takes about 30–60 seconds. The Pinecone free tier supports one index.
> You only need to run this once unless you change your documents.

---

## Step 6 — Test Each Node Individually (Groq)

This is the recommended approach to avoid exhausting Groq's rate limits.
Test one node at a time — each test file makes minimal LLM calls.

### 6a. Test Router (4 LLM calls)
```powershell
python -m pytest tests/test_router.py -v -s
```
✅ Verifies: question classification → correct namespace

### 6b. Test Retriever (0 LLM calls — only embeddings)
```powershell
python -m pytest tests/test_retriever.py -v -s
```
✅ Verifies: Pinecone search returns correct chunks with metadata

### 6c. Test Grader (2 LLM calls per test)
```powershell
python -m pytest tests/test_grader.py -v -s
```
✅ Verifies: relevant docs pass, irrelevant docs are filtered

### 6d. Test Generator (1 LLM call per test)
```powershell
python -m pytest tests/test_generator.py -v -s
```
✅ Verifies: answer is generated, uses context, admits when info is missing

### 6e. Test Checker (1 LLM call per test)
```powershell
python -m pytest tests/test_checker.py -v -s
```
✅ Verifies: grounded answers pass, hallucinated answers are flagged

### Run All Tests at Once (if you want)
```powershell
python -m pytest tests/ -v -s
```

---

## Step 7 — Test the Full Pipeline (End-to-End with Groq)

Run a single question through the complete 5-node graph:

```powershell
python -c "
from src.graph import run_query
result = run_query('How many days of annual leave do employees get?')
print('Answer:', result['generation'])
print('Namespace:', result['doc_type'])
print('Docs retrieved:', len(result['documents']))
"
```

---

## Step 8 — Switch to OpenAI

Once you're happy with Groq testing, switch the LLM provider.
**Only one line changes in .env:**

```
# In .env, change:
LLM_PROVIDER=groq
# → to:
LLM_PROVIDER=openai
```

Save the file. No code changes needed. All nodes read from `get_llm()` in `config.py`.

> **Embeddings are NOT affected by this switch.** They always use HuggingFace all-MiniLM-L6-v2
> (free, local). OpenAI is used only as an LLM when `LLM_PROVIDER=openai`.

---

## Step 9 — Run the Evaluation (15 Questions)

```powershell
python evaluate.py
```

This runs all 15 questions including edge cases through the full pipeline and saves:
- Console: summary table with routing accuracy and answerability scores
- File: `evaluation_report.json` — detailed per-question results

---

## Step 10 — Launch the Streamlit App

```powershell
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

The UI shows:
- Chat interface with history
- Source documents cited per answer
- Namespace the question was routed to
- Retry count (shows how many loops the graph needed)

---

## Project Structure

```
enterprise-rag-bot/
├── .env                    ← your real API keys (gitignored)
├── .env.example            ← template (safe to commit)
├── requirements.txt
├── app.py                  ← Streamlit chat UI
├── evaluate.py             ← 15-question evaluation script
│
├── docs/
│   ├── hr/
│   │   └── hr_policy.txt
│   ├── technical/
│   │   └── tech_docs.txt
│   └── compliance/
│       └── compliance_manual.txt
│
├── src/
│   ├── state.py            ← RAGState TypedDict (flows through all nodes)
│   ├── config.py           ← LLM/Pinecone config + get_llm() / get_embeddings()
│   ├── ingestion.py        ← One-time document loading → chunking → Pinecone
│   ├── graph.py            ← LangGraph StateGraph wiring all 5 nodes
│   └── nodes/
│       ├── router.py       ← Node 1: classify question → namespace
│       ├── retriever.py    ← Node 2: Pinecone similarity search
│       ├── grader.py       ← Node 3: filter irrelevant chunks
│       ├── generator.py    ← Node 4: LLM answer generation
│       └── checker.py      ← Node 5: hallucination check
│
└── tests/
    ├── test_router.py
    ├── test_retriever.py
    ├── test_grader.py
    ├── test_generator.py
    └── test_checker.py
```

---

## Switching LLM — Quick Reference

| What you want | .env setting | Embeddings |
|---|---|---|
| Test with Groq (free) | `LLM_PROVIDER=groq` | HuggingFace (always free) |
| Production with OpenAI | `LLM_PROVIDER=openai` | HuggingFace (always free) |

That's it. Embeddings never change — always HuggingFace, always local, always free.

---

## Common Issues

**`ModuleNotFoundError: No module named 'src'`**
→ Make sure you're running from the project root (`enterprise-rag-bot/`), not inside `src/`.
→ Use `python -m pytest` not `python pytest`.

**`AuthenticationError` from Pinecone/Groq/OpenAI**
→ Check your API keys in `.env`. Make sure there are no spaces around `=`.

**`Index not found` error**
→ You haven't run ingestion yet. Run: `python -m src.ingestion`

**Pinecone `dimension mismatch` error**
→ You likely have an old index from a previous run with `dim=1536` (OpenAI embeddings).
→ Go to https://app.pinecone.io → delete the `enterprise-rag` index → re-run ingestion.
→ The new index must be `dim=384` (HuggingFace all-MiniLM-L6-v2).

**Slow first run of ingestion or tests**
→ The HuggingFace model (~90MB) downloads on first use. Subsequent runs are instant (cached).

**Groq rate limit hit during tests**
→ Run one test file at a time. Wait 60 seconds between files.
→ Or switch to OpenAI for LLM calls: `LLM_PROVIDER=openai` in `.env`.

**PowerShell execution policy error**
→ Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
