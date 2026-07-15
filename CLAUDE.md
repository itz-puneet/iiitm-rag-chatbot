# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **fully local, free** RAG pipeline — ingestion *and* retrieval — for a chatbot
about ABV-IIITM Gwalior (`https://www.iiitm.ac.in`). It crawls the site, extracts
table-aware text, chunks + tags it, embeds it, indexes it in FAISS, and serves
hybrid (semantic + keyword) retrieval with reranking and a cited-answer layer.

**No cloud services or paid APIs anywhere — this is a hard design constraint.** In
particular the whole ML stack deliberately avoids `torch`: embeddings and the
reranker run on **ONNX** (`fastembed`, reusing the `onnxruntime` that RapidOCR
pulls in). A HuggingFace hosted-inference approach was tried and abandoned (its
free tier is deprecated/unreliable — 401/500s). Keep additions offline/free/ONNX.

## The pipeline (this is the architecture)

Seven standalone scripts run in sequence. **Each stage's output is the next
stage's input** — that hand-off is the whole architecture:

```
download_pdfs.py      ──▶ pdfs/         159 curated PDFs
extract_pdfs_local.py ──▶ extracted/    one .md per PDF (table-aware)
chunk_documents.py    ──▶ chunks.jsonl  13,291 chunks + metadata
embed_chunks.py       ──▶ embeddings.npy   13291×384 float32, L2-normalized
build_faiss.py        ──▶ faiss_store/   index.faiss + docstore.jsonl + config.json
hybrid_search.py      ──▶ HybridRetriever (vector + BM25 + RRF + cross-encoder rerank)
answer.py             ──▶ retrieve → grounded prompt → local LLM → cited answer
```

Three couplings between stages are non-obvious and easy to break:

1. **Provenance header.** `extract_pdfs_local.py` writes a header into every `.md`:
   `# <stem>` then `> source: <original>.pdf | method: <digital|ocr>`.
   `chunk_documents.py::load_body()` parses that `> source:` line to recover the
   original PDF name for the `source` metadata field, then strips the header so it
   isn't chunked. If you change the header format in one script, update the parser in
   the other.
2. **Digital-vs-scanned routing.** `extract_pdfs_local.py::is_scanned()` treats a PDF
   as image-only when it averages `< SCANNED_CHARS_PER_PAGE` (90) chars/page, sending
   it to RapidOCR instead of pymupdf4llm. ~26 of the 159 PDFs are scans (Statute, MoUs,
   notifications, a 296-page rules compilation). Plain OCR does **not** preserve table
   grid structure, so scanned tables become loose text — acceptable here because the
   critical fee/credit tables are all in digital PDFs.
3. **Row alignment.** `embeddings.npy` row *i* ↔ `chunks.jsonl` line *i* ↔
   `faiss_store/docstore.jsonl` line *i* ↔ FAISS index id *i*. This positional
   identity is how metadata rejoins vectors (FAISS stores vectors only). If you
   re-chunk, you MUST re-embed and rebuild the store, or the alignment silently
   breaks. `build_faiss.py` guards this with a count check, not a content check.

## Commands

Environment: **Windows / PowerShell**, Python 3.10, virtualenv in `.venv`.

```powershell
# one-time dependency install (there is no requirements.txt)
pip install requests beautifulsoup4 pymupdf4llm rapidocr-onnxruntime `
            langchain-text-splitters fastembed faiss-cpu rank-bm25

# stage 1: crawl + download PDFs (BFS, same-domain, polite delay)
python download_pdfs.py --depth 2 --out ./pdfs

# stage 2: PDF -> Markdown (resumable; --no-ocr to skip scans)
python extract_pdfs_local.py --in ./pdfs --out ./extracted

# stage 3: chunk + tag metadata -> chunks.jsonl
python chunk_documents.py --in ./extracted --out ./chunks.jsonl --size 1000 --overlap 150

# stage 4: embed chunks -> embeddings.npy (fastembed / bge-small-en-v1.5, ONNX)
python embed_chunks.py
python embed_chunks.py --query "how much is the M.Tech hostel fee?"   # search demo

# stage 5: load vectors + metadata into FAISS -> faiss_store/
python build_faiss.py
python build_faiss.py --query "credit requirements" --audience MTech --topic Curriculum

# stage 6: hybrid search (vector + BM25 + RRF, optional cross-encoder rerank)
python hybrid_search.py --query "rule 30.7" --rerank
python hybrid_search.py --query "hostel refund" --mode bm25   # or vector / hybrid

# stage 7: cited answers (needs a local LLM: `ollama pull llama3.2`, or an
#          OpenAI-compatible endpoint via OPENAI_API_KEY + OPENAI_BASE_URL)
python answer.py --q "What is the hostel mess fee for M.Tech students?" --audience MTech
```

**Always prefix console runs with `PYTHONIOENCODING=utf-8`** (bash) or set
`$env:PYTHONIOENCODING="utf-8"` (PowerShell). Extracted text contains `₹` and Hindi
(Devanagari); the default Windows cp1252 console will crash with `UnicodeEncodeError`
on `print`. File writes already force `encoding="utf-8"`.

Stages 2–5 are cheap to re-run; stage 2 is **resumable** (skips PDFs whose `.md`
exists). **After re-running stage 3, always re-run 4 and 5** (row-alignment coupling
above). There is no build step, linter, or test suite; verify changes by running a
stage and eyeballing its summary (e.g. the chunk-by-audience / chunk-by-topic tallies
stage 3 prints, or the `--query` demos in stages 4–6). First run of stages 4/6
downloads the ONNX models (~130 MB embedder, ~90 MB reranker), then works offline.

## Retrieval semantics baked into the metadata

`chunk_documents.py` tags each chunk with `source`, `audience` (list), `topic`,
and `section`. When building the retrieval layer, honor how these were designed:

- **`audience` is a list, not a scalar** (a fee table with per-program columns is
  `["BTech","MTech","MBA"]`). Filter with a *contains* check, never equality.
- **`"General"` means institute-wide** (no program named — ~8.3k of 13.3k chunks:
  hostel, RTI, holidays, common ordinances). A program query must match
  `audience ∈ {<program>, "General"}`, or it will miss shared policies.
- **Audience precedence:** the filename is authoritative for single-program docs
  (`M.Tech-in-CSE-….pdf` → every chunk `MTech`); only multi-program/generic docs get
  per-chunk detection. For those, detection reads the chunk text **plus its Markdown
  heading breadcrumb** (`breadcrumb_at`), so a chunk deep under `SEMESTER-III` still
  inherits its program from the ancestor `## M. Tech. (…)` heading — without the
  breadcrumb, consolidated `All-PG-Programmes-…` chunks mis-tag as `General` and one
  program's content leaks under another's filter.

Classification tuning lives in the `AUDIENCE_KEYWORDS` / `TOPIC_KEYWORDS` dicts at the
top of `chunk_documents.py` — edit there, not in the functions. Matching is via
`_compile`: a **word-boundary, separator-flexible regex** (a space in a keyword
matches any run of space/hyphen/underscore), so `Master-of-Business-Administration`
and `MS_AI_DS` match the space keywords while `\b` blocks substring false positives
(`programs aim` ≠ `ms ai`). Do NOT revert this to plain substring matching or to
normalizing the text (an earlier `_norm` that lower-cased separators silently killed
hyphenated keywords). `topic` is single-label per chunk and keyword-scored; it is
approximate (e.g. `Research` over-fires because the word is common).

## Retrieval + answer layer

- **Hybrid = FAISS (semantic) + BM25 (keyword), fused with Reciprocal Rank Fusion.**
  Vector search alone blurs exact identifiers (`Rule 14.2` ≈ `14.3`); BM25 catches them
  but misses paraphrases. Both retrievers apply the same metadata filter (`matches()`)
  — FAISS via a native `IDSelectorBatch`, BM25 by filtering candidates.
- **BM25 tokenizer keeps dotted identifiers whole** (`hybrid_search.py::tokenize`):
  `Rule 30.7` → `["rule","30.7"]`. A normal tokenizer shatters `30.7` into `30`+`7` and
  destroys the exact match that BM25 exists to provide. Same tokenizer indexes and queries.
- **`k_rrf=15`, not the textbook 60** — deliberate (see the long docstring in
  `HybridRetriever.search`). Our two retrievers are authoritative for disjoint query
  types and their result lists often don't overlap; a large `k` lets a chunk sitting
  mid-list in *both* rankers outrank an exact match that is #1 in one but absent from
  the other. Lowering it keeps each retriever's strong top hits near the top.
- **Reranker is opt-in** (`--rerank`): fuses a larger pool, then reorders with a
  `fastembed` ONNX cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`, no torch). Use it to
  clean up bare-identifier queries where vector search injects a confident-but-wrong #1.
- **`answer.py` LLM backend** is pluggable and free: OpenAI-compatible endpoint if
  `OPENAI_API_KEY` is set (e.g. a free Groq/OpenRouter key or local vLLM), else Ollama
  at `OLLAMA_URL`, else it prints the assembled grounded prompt so the RAG is still
  auditable with no LLM running. Retrieved sources are always printed as `[n]` citations.

## Corpus note

`pdfs/` was hand-curated from 421 crawled files down to 159 by removing: exact
duplicates, student magazines, governance minutes & financial/audit reports,
procurement/tenders, blank forms, staff-recruitment churn, and superseded old editions
(kept only the newest version where one clearly supersedes another). Keepers are
durable, question-worthy college info: programs/curricula, ordinances, fee structures,
academic calendars, admissions, hostel, scholarships, placements, brochures, policies.
Preserve that intent if regenerating the corpus.
