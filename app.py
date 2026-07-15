"""
Minimal chat UI for the ABV-IIITM RAG system (stage 8).

Run:  pip install streamlit
      streamlit run app.py
For generated answers, configure an LLM in .env (OPENAI_API_KEY + OPENAI_BASE_URL
+ OPENAI_MODEL for a free Groq key, or run Ollama locally).
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
    return HybridRetriever()          # load FAISS + models ONCE, reuse across turns


PROGRAMS = ["", "BTech", "MTech", "MBA", "MS", "PhD", "IPG"]
TOPICS = ["", "Fees", "Curriculum", "Admissions", "Hostel", "Scholarship",
          "Examination", "Ordinance/Rules", "Placement", "AcademicCalendar",
          "Library", "Research"]
audience = st.sidebar.selectbox("Filter by program", PROGRAMS)
topic = st.sidebar.selectbox("Filter by topic", TOPICS)
rerank = st.sidebar.checkbox("Rerank (higher quality, slower)", True)

st.session_state.setdefault("messages", [])
for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

if q := st.chat_input("Ask about programs, fees, rules, hostel…"):
    st.session_state.messages.append({"role": "user", "content": q})
    st.chat_message("user").markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching college documents…"):
            res = answer(q, audience=audience, topic=topic,
                         rerank=rerank, retriever=get_retriever())
        reply = (res["answer"] if res["llm_ok"]
                 else "⚠️ Unable to reach the server. Please try again later.")
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
