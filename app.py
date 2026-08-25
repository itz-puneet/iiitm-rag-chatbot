"""
Minimal chat UI for the ABV-IIITM RAG system (stage 8).

Run:  pip install streamlit
      streamlit run app.py
For generated answers, configure an LLM in .env:
  - Free Google Gemini (Recommended): GEMINI_API_KEY=...
  - Free Groq / OpenRouter: OPENAI_API_KEY=... + OPENAI_BASE_URL=... + OPENAI_MODEL=...
  - Local Ollama: OLLAMA_URL=http://localhost:11434 (with `ollama pull llama3.2`)
"""
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from answer import answer, HybridRetriever

# Re-read .env on every rerun so edits to the key/model take effect immediately,
# without restarting Streamlit (override=True refreshes already-loaded values).
load_dotenv(Path(__file__).with_name(".env"), override=True)

st.set_page_config(page_title="ABV-IIITM Assistant", page_icon="🎓")
st.title("🎓 ABV-IIITM College Assistant")


@st.cache_resource
def get_retriever():
    try:
        return HybridRetriever()          # load FAISS + models ONCE, reuse across turns
    except FileNotFoundError:
        return None


PROGRAMS = ["", "BTech", "MTech", "MBA", "MS", "PhD", "IPG"]
TOPICS = ["", "Fees", "Curriculum", "Admissions", "Hostel", "Scholarship",
          "Examination", "Ordinance/Rules", "Placement", "AcademicCalendar",
          "Library", "Research"]
audience = st.sidebar.selectbox("Filter by program", PROGRAMS)
topic = st.sidebar.selectbox("Filter by topic", TOPICS)
rerank = st.sidebar.checkbox("Rerank (higher quality, slower)", True)

retriever = get_retriever()
if retriever is None:
    st.sidebar.warning("⚠️ **Knowledge base index not built.**")
    st.info(
        "👋 Welcome! Your Google Gemini API key is configured and working.\n\n"
        "To enable document retrieval and citations, build the college knowledge base by running:\n"
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
    st.chat_message(m["role"]).markdown(m["content"])

if q := st.chat_input("Ask about programs, fees, rules, hostel…"):
    st.session_state.messages.append({"role": "user", "content": q})
    st.chat_message("user").markdown(q)
    with st.chat_message("assistant"):
        if retriever is None:
            reply = (
                "⚠️ The document index (`faiss_store/`) has not been built yet.\n\n"
                "Please run the pipeline stages (1-5) as shown above so I can search and cite the official PDFs."
            )
        else:
            with st.spinner("Searching college documents…"):
                res = answer(q, audience=audience, topic=topic,
                             rerank=rerank, retriever=retriever)
            if res["llm_ok"]:
                reply = res["answer"]
            else:
                err = res.get("error", "Unknown error")
                reply = f"⚠️ **LLM Error**: {err}\n\n*Please verify your `.env` configuration.*"
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

