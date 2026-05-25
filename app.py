"""
Enterprise RAG Bot — Streamlit UI
------------------------------------
Run: streamlit run app.py
"""
import streamlit as st
from src.config import get_active_provider
from src.graph import run_query

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Enterprise RAG Bot",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏢 Enterprise RAG Bot")
    st.caption("Powered by LangGraph + Pinecone")

    st.divider()

    st.subheader("📚 Knowledge Base")
    st.markdown("""
- 📋 **HR Policies** — leave, benefits, remote work, reviews
- 📘 **Technical Docs** — APIs, error codes, deployments
- ⚖️ **Compliance** — GDPR, security, incident response
    """)

    st.divider()

    st.subheader("🔷 Pipeline Nodes")
    st.markdown("""
1. **Router** — classifies your question
2. **Retriever** — fetches chunks from Pinecone
3. **Grader** — filters for relevance
4. **Generator** — creates the answer
5. **Checker** — verifies no hallucinations
    """)

    st.divider()

    st.subheader("⚙️ Active LLM")
    provider = get_active_provider()
    st.info(f"🤖 {provider}")
    st.caption("Change `LLM_PROVIDER` in `.env` to switch between Groq and OpenAI.")

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.caption("💡 **Tip:** Ask edge-case questions to see how the pipeline handles them!")

# ─── Main Chat Interface ───────────────────────────────────────────────────────

st.header("Ask your Enterprise Knowledge Base")
st.caption("Try: leave policies, API authentication, GDPR requirements, or anything that spans multiple topics.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant":
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if msg.get("sources"):
                    with st.expander("📎 Source Documents"):
                        for src in msg["sources"]:
                            st.caption(f"• {src}")
            with col2:
                if msg.get("doc_type"):
                    st.metric("Namespace", msg["doc_type"].upper())
            with col3:
                if msg.get("retry_count") is not None:
                    st.metric("Retries", msg["retry_count"])

# ─── Chat Input ───────────────────────────────────────────────────────────────

if question := st.chat_input("Ask about HR, technical docs, or compliance..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Run pipeline
    with st.chat_message("assistant"):
        with st.spinner("Running RAG pipeline..."):
            try:
                result = run_query(question)

                answer      = result.get("generation", "")
                doc_type    = result.get("doc_type", "unknown")
                retry_count = result.get("retry_count", 0)
                sources     = sorted(set(
                    doc.metadata.get("source", "unknown")
                    for doc in result.get("documents", [])
                ))

                # Handle empty answer
                if not answer:
                    answer = "I couldn't generate an answer. The question may be out of scope."

                st.write(answer)

                # Metadata row
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    if sources:
                        with st.expander("📎 Source Documents"):
                            for src in sources:
                                st.caption(f"• {src}")
                    else:
                        st.caption("No source documents retrieved.")
                with col2:
                    st.metric("Namespace", doc_type.upper())
                with col3:
                    st.metric("Retries", retry_count)

                # Save to history
                st.session_state.messages.append({
                    "role":        "assistant",
                    "content":     answer,
                    "sources":     sources,
                    "doc_type":    doc_type,
                    "retry_count": retry_count,
                })

            except Exception as e:
                error_msg = f"Pipeline error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", "content": error_msg,
                })

# ─── Example Questions ────────────────────────────────────────────────────────

with st.expander("💡 Example Questions to Try"):
    examples = {
        "HR":         ["How many annual leave days do I get?",
                       "What is the remote work policy?",
                       "How does the 401k matching work?"],
        "Technical":  ["How do I authenticate with the API?",
                       "What does a 429 error mean?",
                       "What is the rate limit for the standard tier?"],
        "Compliance": ["What are the GDPR data retention requirements?",
                       "How quickly must we report a data breach?",
                       "What is the vendor approval process?"],
        "Edge Cases": ["What's the overlap between leave policy and data privacy for HR data?",
                       "Can employees work from Mars?",
                       "What is our company revenue?"],
    }
    for category, questions in examples.items():
        st.markdown(f"**{category}**")
        for q in questions:
            st.caption(f"• {q}")
