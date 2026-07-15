"""
Stage 7: the RAG answer layer — retrieve, then generate a cited answer.

Pipeline: HybridRetriever (vector + BM25 + rerank)  ->  numbered context  ->
grounded prompt  ->  local/free LLM  ->  answer with [n] citations.

LLM backend (free, in priority order):
  1. OpenAI-compatible endpoint if OPENAI_API_KEY is set (works with a free Groq
     or OpenRouter key, or a local vLLM/llama.cpp server) — set OPENAI_BASE_URL.
  2. Ollama at OLLAMA_URL (default http://localhost:11434) — fully local & free
     (`ollama pull llama3.2`).
  3. If neither is reachable, the assembled prompt is printed so you can see the
     retrieval + grounding work and drop in any LLM.

The retrieved sources are ALWAYS printed, so answers stay auditable.

Setup:
    pip install requests            # already a project dependency
    # then EITHER:  ollama pull llama3.2                       (local; default model)
    #        OR:    $env:OPENAI_API_KEY="..."; $env:OPENAI_BASE_URL="https://api.groq.com/openai/v1"
    #               and pass a real model id, e.g.  --model llama-3.3-70b-versatile

Usage:
    python answer.py --q "What is the hostel mess fee for M.Tech students?"
    python answer.py --q "What does rule 30.7 cover?" --audience MTech
    python answer.py --q "..." --model llama3.2 --show-context
"""

import argparse
import os
from pathlib import Path

import requests

from hybrid_search import HybridRetriever

# Load OPENAI_API_KEY / OPENAI_BASE_URL / OLLAMA_URL from a .env file next to this
# script. This makes config work regardless of which shell (or none) launches the
# CLI or Streamlit — the #1 cause of "env vars not visible to the app" on Windows.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

SYSTEM = (
    "You are the ABV-IIITM Gwalior college information assistant. Answer the "
    "question using ONLY the numbered context sources provided. Cite every fact "
    "with its source marker like [1] or [2]. Quote fee amounts, credit counts, "
    "rule/form numbers, and dates EXACTLY as written. If the answer is not in the "
    "context, say: 'I don't have that information in the college documents.' Do "
    "not invent anything."
)


def build_context(hits) -> str:
    blocks = []
    for n, h in enumerate(hits, 1):
        m = h["metadata"]
        header = (f"[{n}] source: {m['source']} | section: {m.get('section', '')} "
                  f"| audience: {m['audience']} | topic: {m['topic']}")
        blocks.append(f"{header}\n{h['text']}")
    return "\n\n".join(blocks)


def build_prompt(question: str, context: str) -> str:
    return (f"Context sources:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer (cite sources as [n]):")


def backend() -> str:
    """Which LLM backend generate() will use, given the environment."""
    return "openai" if os.environ.get("OPENAI_API_KEY") else "ollama"


def _check(resp, where: str):
    """Raise with the response BODY included — API errors (bad model, quota, bad
    key) live in the body, not in the bare status line raise_for_status() gives."""
    if resp.status_code >= 400:
        raise RuntimeError(f"{where} returned {resp.status_code}: {resp.text[:400]}")


def generate(system: str, prompt: str, model: str) -> str:
    """Call the configured free LLM backend; raise (with body) if it fails.

    The model name is backend-specific, so it is defaulted PER backend: Ollama
    falls back to 'llama3.2'; the OpenAI-compatible path has no safe default
    (Groq/OpenRouter use ids like 'llama-3.3-70b-versatile'), so it requires
    --model or OPENAI_MODEL and fails loudly rather than sending a bad id.
    """
    if backend() == "openai":  # Groq / OpenRouter / local vLLM
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = model or os.environ.get("OPENAI_MODEL")
        if not model:
            raise RuntimeError(
                "OPENAI_API_KEY is set but no model given. Pass --model (e.g. "
                "'llama-3.3-70b-versatile' for Groq) or set OPENAI_MODEL.")
        resp = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={"model": model, "temperature": 0, "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}]},
            timeout=120)
        _check(resp, f"OpenAI-compatible endpoint ({base})")
        return resp.json()["choices"][0]["message"]["content"]

    # Ollama (fully local)
    base = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    resp = requests.post(
        f"{base}/api/chat",
        json={"model": model or "llama3.2", "stream": False, "options": {"temperature": 0},
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": prompt}]},
        timeout=120)
    _check(resp, "Ollama")
    return resp.json()["message"]["content"]


NO_CONTEXT_MSG = "I don't have that information in the college documents."


def answer(question, k=5, audience="", topic="", model="", rerank=True,
           retriever=None):
    r = retriever or HybridRetriever()
    hits = r.search(question, k=k, audience=audience, topic=topic, rerank=rerank)
    # No retrieved context -> refuse deterministically instead of asking the LLM
    # to answer from an empty context (which invites ungrounded, uncitable output).
    if not hits:
        return {"answer": NO_CONTEXT_MSG, "hits": [], "prompt": None,
                "llm_ok": True, "no_context": True}
    context = build_context(hits)
    prompt = build_prompt(question, context)
    try:
        text = generate(SYSTEM, prompt, model)
        return {"answer": text, "hits": hits, "prompt": prompt, "llm_ok": True}
    except Exception as exc:
        return {"answer": None, "hits": hits, "prompt": prompt,
                "llm_ok": False, "error": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG answer layer (retrieve -> cite -> generate).")
    ap.add_argument("--q", "--question", dest="question", required=True)
    ap.add_argument("--audience", default="")
    ap.add_argument("--topic", default="")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default="", help="model name (default: Ollama 'llama3.2'; "
                    "for OpenAI-compatible pass a real id or set OPENAI_MODEL)")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--show-context", action="store_true", help="print the retrieved context")
    args = ap.parse_args()

    res = answer(args.question, k=args.k, audience=args.audience, topic=args.topic,
                 model=args.model, rerank=not args.no_rerank)

    print(f'\nQ: {args.question}\n' + "=" * 70)
    if res.get("no_context"):
        print(res["answer"])
        print("\n(No chunks matched the query/filters — nothing retrieved to cite.)")
        return
    if res["llm_ok"]:
        print(res["answer"])
    else:
        # Tailor remediation to the backend that was actually attempted.
        print(f"[LLM call failed: {res['error']}]")
        if backend() == "openai":
            print("The OpenAI-compatible endpoint (OPENAI_API_KEY is set) rejected the call.")
            print("Check OPENAI_BASE_URL, the key, and pass a valid --model / OPENAI_MODEL.")
        else:
            print("No local LLM reachable. Start one (free):")
            print(f"    ollama pull {args.model or 'llama3.2'} && ollama serve")
            print("    or set OPENAI_API_KEY + OPENAI_BASE_URL (e.g. a free Groq key).")
        print("\n--- Assembled RAG prompt (retrieval + grounding is working) ---")
        print(res["prompt"][:1500] + ("..." if len(res["prompt"]) > 1500 else ""))

    print("\n--- Retrieved sources (citations) ---")
    for n, h in enumerate(res["hits"], 1):
        m = h["metadata"]
        print(f"[{n}] {m['source']} | {m.get('section','')} | audience={m['audience']} topic={m['topic']}")
        if args.show_context:
            print(f"    {h['text'][:200].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
