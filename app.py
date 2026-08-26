"""
Minimal chat UI for the ABV-IIITM RAG system (stage 8).

Run:  pip install streamlit
      streamlit run app.py
For generated answers, configure an LLM in .env:
  - Free Google Gemini (Recommended): GEMINI_API_KEY=...
  - Free Groq / OpenRouter: OPENAI_API_KEY=... + OPENAI_BASE_URL=... + OPENAI_MODEL=...
  - Local Ollama: OLLAMA_URL=http://localhost:11434 (with `ollama pull llama3.2`)
"""
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from answer import answer, HybridRetriever

# Re-read .env on every rerun so edits to the key/model take effect immediately
load_dotenv(Path(__file__).with_name(".env"), override=True)

# Sync Streamlit Cloud secrets to environment variables
try:
    for k, v in st.secrets.items():
        if isinstance(v, str) and k not in os.environ:
            os.environ[k] = v
except Exception:
    pass

st.set_page_config(page_title="ABV-IIITM Assistant", page_icon="🎓", layout="wide")
st.title("🎓 ABV-IIITM College Assistant")
st.caption("AI-powered RAG chatbot grounded in official ABV-IIITM Gwalior institutional documents, ordinances, and fee structures.")



@st.cache_resource
def get_retriever():
    try:
        return HybridRetriever()          # load FAISS + models ONCE, reuse across turns
    except FileNotFoundError:
        return None


# Sidebar filters and settings
with st.sidebar:
    st.header("🔍 Search Filters")
    PROGRAMS = ["", "BTech", "MTech", "MBA", "MS", "PhD", "IPG"]
    TOPICS = ["", "Fees", "Curriculum", "Admissions", "Hostel", "Scholarship",
              "Examination", "Ordinance/Rules", "Placement", "AcademicCalendar",
              "Library", "Research"]
    audience = st.selectbox("Filter by program", PROGRAMS)
    topic = st.selectbox("Filter by topic", TOPICS)
    rerank = st.checkbox("Cross-encoder Rerank (higher precision)", True)

    st.divider()
    st.header("💡 Example Questions")
    sample_queries = [
        "What is the fee structure for B.Tech?",
        "What are the hostel rules and curfew timings?",
        "What is the minimum credit requirement for M.Tech thesis?",
        "What scholarships are available for students?",
        "What is the grading system and CGPA criteria?"
    ]
    for sq in sample_queries:
        if st.button(sq, use_container_width=True):
            st.session_state["queued_query"] = sq

    st.divider()
    with st.expander("⚙️ API Key Configuration (Optional)"):
        user_key = st.text_input("Gemini or Groq API Key", type="password", help="If not set in secrets/.env, you can paste your key here.")
        if user_key:
            if user_key.startswith("gsk_"):
                os.environ["OPENAI_API_KEY"] = user_key
                os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
                os.environ["OPENAI_MODEL"] = "llama-3.3-70b-versatile"
                os.environ["LLM_BACKEND"] = "openai"
            else:
                os.environ["GEMINI_API_KEY"] = user_key
                os.environ["LLM_BACKEND"] = "gemini"

    st.divider()
    st.caption("📊 **System Specs**\n- **Indexed Chunks**: 12,511\n- **Embeddings**: BAAI/bge-small-en-v1.5\n- **Retrieval**: Hybrid (BM25 + FAISS Dense) with RRF")

retriever = get_retriever()
if retriever is None:
    st.sidebar.warning("⚠️ **Knowledge base index not found.**")
    st.info(
        "👋 Welcome! Build the college knowledge base by running:\n"
        "```bash\n"
        "python download_pdfs.py --depth 2 --out ./pdfs\n"
        "python extract_pdfs_local.py\n"
        "python chunk_documents.py\n"
        "python embed_chunks.py\n"
        "python build_faiss.py\n"
        "```"
    )

st.session_state.setdefault("messages", [])
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("📚 View Cited Sources", expanded=False):
                for s in m["sources"]:
                    st.markdown(f"**[{s['num']}] {s['source']}**  \n*Section: {s.get('section', 'N/A')} | Audience: {s.get('audience')} | Topic: {s.get('topic')}*  \n> {s['text']}")

# Handle queued queries from example buttons
query_input = st.chat_input("Ask about programs, fees, rules, hostel, placements…")
if "queued_query" in st.session_state and st.session_state["queued_query"]:
    query_input = st.session_state.pop("queued_query")

if query_input:
    st.session_state.messages.append({"role": "user", "content": query_input})
    with st.chat_message("user"):
        st.markdown(query_input)
    
    with st.chat_message("assistant"):
        if retriever is None:
            reply = (
                "⚠️ The document index (`faiss_store/`) has not been built yet.\n\n"
                "Please ensure `faiss_store/` exists in the project root."
            )
            sources = []
            st.markdown(reply)
        else:
            with st.spinner("🔍 Searching college documents & synthesizing answer…"):
                res = answer(query_input, audience=audience, topic=topic,
                             rerank=rerank, retriever=retriever)
            
            sources = []
            if res["llm_ok"]:
                reply = res["answer"]
                if res.get("hits"):
                    for n, h in enumerate(res["hits"], 1):
                        m = h["metadata"]
                        sources.append({
                            "num": n,
                            "source": m.get("source", "PDF Document"),
                            "section": m.get("section", ""),
                            "audience": m.get("audience", []),
                            "topic": m.get("topic", ""),
                            "text": h.get("text", "")[:400] + "..."
                        })
            else:
                err = res.get("error", "Unknown error")
                reply = f"⚠️ **LLM Generation Error**: {err}\n\n*Please verify your API key in `.env` or Streamlit Secrets.*"
            
            st.markdown(reply)
            if sources:
                with st.expander("📚 View Cited Sources", expanded=False):
                    for s in sources:
                        st.markdown(f"**[{s['num']}] {s['source']}**  \n*Section: {s.get('section', 'N/A')} | Audience: {s.get('audience')} | Topic: {s.get('topic')}*  \n> {s['text']}")
        
        st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})


