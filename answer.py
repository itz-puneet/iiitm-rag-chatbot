"""
Stage 7: the RAG answer layer — retrieve, then generate a cited answer.

Pipeline: HybridRetriever (vector + BM25 + rerank)  ->  numbered context  ->
grounded prompt  ->  local/free LLM  ->  answer with [n] citations.

LLM backends (free, in priority order):
  1. Google Gemini API (Recommended) if GEMINI_API_KEY or GOOGLE_API_KEY is set.
     Get a free key at https://aistudio.google.com/ (default model: gemini-2.5-flash).
  2. OpenAI-compatible endpoint if OPENAI_API_KEY is set (works with a free Groq
     or OpenRouter key, or a local vLLM/llama.cpp server) — set OPENAI_BASE_URL.
  3. Ollama at OLLAMA_URL (default http://localhost:11434) — fully local & free
     (`ollama pull llama3.2`).
  4. If none is reachable, the assembled prompt is printed so you can see the
     retrieval + grounding work and drop in any LLM.

The retrieved sources are ALWAYS printed, so answers stay auditable.

Setup:
    pip install requests            # already a project dependency
    # Option A (Recommended): set GEMINI_API_KEY in .env (free from https://aistudio.google.com/)
    # Option B:               set OPENAI_API_KEY + OPENAI_BASE_URL in .env (e.g. Groq)
    # Option C:               ollama pull llama3.2 (local)

Usage:
    python answer.py --q "What is the hostel mess fee for M.Tech students?"
    python answer.py --q "What does rule 30.7 cover?" --audience MTech
    python answer.py --q "..." --backend gemini --model gemini-2.5-flash --show-context
"""

import argparse
import os
from pathlib import Path

import requests

from hybrid_search import HybridRetriever

# Load GEMINI_API_KEY / OPENAI_API_KEY / OPENAI_BASE_URL / OLLAMA_URL from a .env
# file next to this script. This makes config work regardless of which shell (or none)
# launches the CLI or Streamlit — the #1 cause of "env vars not visible to the app" on Windows.
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


def backend(preferred: str = "") -> str:
    """Which LLM backend generate() will use, given the environment and optional preference."""
    if preferred:
        return preferred.lower()
    explicit = os.environ.get("LLM_BACKEND")
    if explicit:
        return explicit.lower()
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


def _check(resp, where: str):
    """Raise with the response BODY included — API errors (bad model, quota, bad
    key) live in the body, not in the bare status line raise_for_status() gives."""
    if resp.status_code >= 400:
        raise RuntimeError(f"{where} returned {resp.status_code}: {resp.text[:400]}")


def generate(system: str, prompt: str, model: str = "", backend_choice: str = "") -> str:
    """Call the configured free LLM backend; raise (with body) if it fails.

    Backends supported:
      - 'gemini': Google Gemini API via GEMINI_API_KEY or GOOGLE_API_KEY. Default model 'gemini-2.5-flash'.
      - 'openai': Groq / OpenRouter / OpenAI-compatible endpoint via OPENAI_API_KEY.
      - 'ollama': Local Ollama instance (default http://localhost:11434, model 'llama3.2').
    """
    b = backend(backend_choice)

    # 1. Google Gemini API (Recommended free tier)
    if b == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "Gemini backend selected but neither GEMINI_API_KEY nor GOOGLE_API_KEY is set. "
                "Get a free API key at https://aistudio.google.com/ and add GEMINI_API_KEY to your .env file."
            )
        # Build list of models to try in order of preference with automatic fallback
        candidate_models = []
        preferred_model = model or os.environ.get("GEMINI_MODEL")
        if preferred_model:
            candidate_models.append(preferred_model)
        
        # Highly available fallback models
        fallbacks = ["gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-flash-lite-latest", "gemini-3.1-flash-lite"]
        for fb in fallbacks:
            if fb not in candidate_models and f"models/{fb}" not in candidate_models:
                candidate_models.append(fb)

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": key
        }
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.0
            }
        }
        if system:
            payload["system_instruction"] = {
                "parts": [{"text": system}]
            }

        last_error = None
        for cand in candidate_models:
            model_clean = cand[7:] if cand.startswith("models/") else cand
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_clean}:generateContent"
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code in (503, 429, 404):
                    last_error = f"{model_clean} (HTTP {resp.status_code}: {resp.text[:150]})"
                    continue
                _check(resp, f"Google Gemini API ({model_clean})")
                data = resp.json()
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    if "promptFeedback" in data and "blockReason" in data["promptFeedback"]:
                        raise RuntimeError(f"Gemini API blocked response: {data['promptFeedback']}")
                    raise RuntimeError(f"Unexpected response format from Gemini API: {data}")
            except Exception as e:
                last_error = str(e)
                if any(err_code in str(e) for err_code in ["503", "429", "404", "high demand"]):
                    continue
                raise

        raise RuntimeError(f"All Gemini models unavailable due to high demand. Last error: {last_error}")


    # 2. OpenAI-compatible endpoint (Groq, OpenRouter, local vLLM, etc.)
    if b == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = model or os.environ.get("OPENAI_MODEL")
        if not model:
            if "groq.com" in base:
                model = "llama-3.3-70b-versatile"
            else:
                raise RuntimeError(
                    "OPENAI_API_KEY is set but no model given. Pass --model (e.g. "
                    "'llama-3.3-70b-versatile' for Groq) or set OPENAI_MODEL in .env.")
        resp = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={"model": model, "temperature": 0, "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}]},
            timeout=120)
        _check(resp, f"OpenAI-compatible endpoint ({base})")
        return resp.json()["choices"][0]["message"]["content"]

    # 3. Ollama (fully local & offline)
    base = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    resp = requests.post(
        f"{base}/api/chat",
        json={"model": model or "llama3.2", "stream": False, "options": {"temperature": 0},
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": prompt}]},
        timeout=120)
    _check(resp, f"Ollama ({base})")
    return resp.json()["message"]["content"]


NO_CONTEXT_MSG = "I don't have that information in the college documents."


def answer(question, k=5, audience="", topic="", model="", backend_choice="", rerank=True,
           retriever=None):
    r = retriever or HybridRetriever()
    hits = r.search(question, k=k, audience=audience, topic=topic, rerank=rerank)
    # No retrieved context -> refuse deterministically instead of asking the LLM
    # to answer from an empty context (which invites ungrounded, uncitable output).
    if not hits:
        return {"answer": NO_CONTEXT_MSG, "hits": [], "prompt": None,
                "llm_ok": True, "no_context": True, "backend": backend(backend_choice)}
    context = build_context(hits)
    prompt = build_prompt(question, context)
    try:
        text = generate(SYSTEM, prompt, model=model, backend_choice=backend_choice)
        return {"answer": text, "hits": hits, "prompt": prompt, "llm_ok": True,
                "backend": backend(backend_choice)}
    except Exception as exc:
        return {"answer": None, "hits": hits, "prompt": prompt,
                "llm_ok": False, "error": str(exc), "backend": backend(backend_choice)}


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG answer layer (retrieve -> cite -> generate).")
    ap.add_argument("--q", "--question", dest="question", required=True)
    ap.add_argument("--audience", default="")
    ap.add_argument("--topic", default="")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--backend", default="", choices=["", "gemini", "openai", "ollama"],
                    help="LLM backend (gemini, openai, ollama; default: auto-detected)")
    ap.add_argument("--model", default="", help="model name (default: Gemini 'gemini-2.5-flash'; "
                    "Groq 'llama-3.3-70b-versatile'; Ollama 'llama3.2')")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--show-context", action="store_true", help="print the retrieved context")
    args = ap.parse_args()

    res = answer(args.question, k=args.k, audience=args.audience, topic=args.topic,
                 model=args.model, backend_choice=args.backend, rerank=not args.no_rerank)

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
        b = res.get("backend", backend(args.backend))
        if b == "gemini":
            print("The Google Gemini API rejected the call.")
            print("Check your GEMINI_API_KEY in .env (get a free key at https://aistudio.google.com/) and GEMINI_MODEL.")
        elif b == "openai":
            print("The OpenAI-compatible endpoint (OPENAI_API_KEY is set) rejected the call.")
            print("Check OPENAI_BASE_URL, the key, and pass a valid --model / OPENAI_MODEL.")
        else:
            print("No LLM backend reachable. Free options:")
            print("    1. Set GEMINI_API_KEY in .env (free key at https://aistudio.google.com/)")
            print("    2. Set OPENAI_API_KEY + OPENAI_BASE_URL in .env (e.g. a free Groq key from https://console.groq.com/)")
            print(f"    3. Start local Ollama: ollama pull {args.model or 'llama3.2'} && ollama serve")
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
