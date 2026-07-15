<div align="center">

# 🎓 ABV-IIITM College RAG Chatbot

**Ask natural-language questions about ABV-IIITM Gwalior — programs, fees, curricula,
ordinances, hostel, scholarships — and get answers cited from the institute's own PDFs.**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Free & Local](https://img.shields.io/badge/100%25-Free%20%26%20Local-2ea44f)
![ML](https://img.shields.io/badge/ML-ONNX%20·%20no%20torch-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

A complete **Retrieval-Augmented Generation** pipeline — crawl → extract → chunk →
embed → index → retrieve → answer — that runs **offline on CPU with no `torch`, no
paid services, and no API keys** for the core pipeline. Embeddings and reranking run
on ONNX. Only the final answer step calls an LLM, and that can be a **free** hosted
model (Groq) or a **local** one (Ollama).

## 💬 Example

```text
Q:  What is the hostel mess fee for M.Tech students?
A:  The hostel mess fee for M.Tech students is ₹20,000.
    └─ source: Fees-Details-for-the-Students-2025-batch-…-2026.pdf
```

## ✨ Features

- **📊 Table-aware ingestion** — fee structures and credit tables survive as real
  Markdown tables; scanned PDFs are OCR'd automatically.
- **🏷️ Metadata-filtered retrieval** — every chunk is tagged with `audience`
  (BTech · MTech · MBA · MS · PhD · IPG) and `topic` (Fees · Curriculum · Hostel · …),
  so a query can be scoped to exactly "M.Tech fees".
- **🔀 Hybrid search** — semantic vectors (FAISS) **+** BM25 keywords, fused with
  Reciprocal Rank Fusion → finds both paraphrased questions **and** exact identifiers
  like `Rule 30.7`.
- **🎯 Cross-encoder reranking** (ONNX, opt-in) for top-result accuracy.
- **🔎 Grounded, cited answers** — every answer traces back to its source documents.
- **🖥️ Clean Streamlit chat UI** with program/topic filters.

## 🏗️ How it works

Eight standalone stages — each stage's output feeds the next:

```text
1. download_pdfs.py        →  pdfs/           crawl the site + download PDFs
2. extract_pdfs_local.py   →  extracted/      PDF → Markdown (pymupdf4llm + RapidOCR)
3. chunk_documents.py      →  chunks.jsonl    recursive chunking + metadata tags
4. embed_chunks.py         →  embeddings.npy  embeddings (fastembed · bge-small-en-v1.5)
5. build_faiss.py          →  faiss_store/    FAISS index + docstore + config
6. hybrid_search.py        →  HybridRetriever (vector + BM25 + RRF + rerank)
7. answer.py               →  retrieve → grounded prompt → LLM → cited answer
8. app.py                  →  Streamlit chat UI
```

## 🧰 Tech stack (all free)

| Stage | Library |
|---|---|
| Crawl | `requests` · `beautifulsoup4` |
| Extract / OCR | `pymupdf4llm` · `rapidocr-onnxruntime` |
| Chunk | `langchain-text-splitters` |
| Embed / rerank | `fastembed` *(ONNX — no torch)* |
| Vector index | `faiss-cpu` |
| Keyword search | `rank-bm25` |
| Answer LLM | Groq *(free tier)* or Ollama *(local)* |
| UI | `streamlit` |

## ⚡ Quickstart

```bash
git clone https://github.com/itz-puneet/iiitm-rag-chatbot.git
cd iiitm-rag-chatbot

python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

cp .env.example .env              # then paste your free Groq key into .env
```

Build the knowledge base (once, in order), then launch the app:

```bash
python download_pdfs.py --depth 2 --out ./pdfs
python extract_pdfs_local.py
python chunk_documents.py
python embed_chunks.py
python build_faiss.py

streamlit run app.py
```

> 💡 The first embed/search run downloads the ONNX models (~130 MB embedder, ~90 MB
> reranker) once, then works fully offline. On Windows, prefix console runs with
> `PYTHONIOENCODING=utf-8` — the documents contain `₹` and Devanagari.

## 🔍 Usage (CLI)

```bash
# hybrid retrieval (add --rerank for the cross-encoder pass)
python hybrid_search.py --query "rule 30.7" --rerank

# a full cited answer, scoped with metadata filters
python answer.py --q "What is the hostel mess fee for M.Tech students?" --audience MTech --topic Fees
```

## ⚙️ Configuration (`.env`)

```ini
# Option A — free hosted LLM (Groq, OpenAI-compatible)
OPENAI_API_KEY=gsk_your_key
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.3-70b-versatile
```

Or leave those unset and run a local model with [Ollama](https://ollama.com):
`ollama pull llama3.2`.

## 📂 Project structure

```text
download_pdfs.py         # 1  crawl + download
extract_pdfs_local.py    # 2  PDF → Markdown (+ OCR)
chunk_documents.py       # 3  chunk + tag metadata
embed_chunks.py          # 4  embeddings
build_faiss.py           # 5  FAISS vector store
hybrid_search.py         # 6  hybrid retrieval + rerank
answer.py                # 7  RAG answer layer
app.py                   # 8  Streamlit UI
requirements.txt · .env.example · CLAUDE.md
```

## 📝 Notes

- **Design constraint:** the whole stack stays free and local, deliberately avoiding
  `torch` (ONNX via `fastembed`). Only the final answer step optionally calls a hosted
  LLM (Groq's free tier).
- **Data is not committed** — `pdfs/`, `extracted/`, `chunks.jsonl`, `embeddings.npy`,
  and `faiss_store/` are large and reproducible; rebuild them with the pipeline above.
- **Retrieval semantics:** `audience` is a *list* (use a contains filter), and
  `"General"` means institute-wide — a program query matches `audience ∈ {<program>, "General"}`.
- Built for **ABV-IIITM Gwalior**, but adaptable to any institution by re-pointing the
  crawler and re-running the pipeline.

<div align="center">
<sub>Built with a fully free, local RAG stack · embeddings & reranking on ONNX</sub>
</div>
