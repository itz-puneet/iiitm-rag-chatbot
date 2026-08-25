"""
Stage 4: turn chunks.jsonl into semantic vectors (FREE, local, open-source).

Uses fastembed (Qdrant) with an open-source HuggingFace embedding model
(default: BAAI/bge-small-en-v1.5 — 384-dim). fastembed runs the model via ONNX
Runtime, so there is NO torch and NO API token: the model downloads once (~130MB)
from HuggingFace, then everything runs offline on CPU.

Output (row i aligns 1:1 with line i of chunks.jsonl):
    embeddings.npy        float32 matrix [n_chunks, dim], L2-normalized
    embeddings_meta.json  {model, dim, count, normalized, source}

Because vectors are L2-normalized, cosine similarity == dot product, so semantic
search is a single matrix-vector multiply (see --query) — no vector DB needed to
prove "search by meaning" works.

Setup:
    pip install fastembed

Usage:
    python embed_chunks.py                                   # build embeddings.npy
    python embed_chunks.py --query "how much is the M.Tech hostel fee?"
    python embed_chunks.py --model sentence-transformers/all-MiniLM-L6-v2  # swap model
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding


def load_chunks(path: Path):
    texts, metas = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                texts.append(row["text"])
                metas.append(row["metadata"])
    return texts, metas


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype="float32")
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed chunks.jsonl locally with fastembed.")
    ap.add_argument("--in", dest="in_file", default="./chunks.jsonl")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out-emb", default="./embeddings.npy")
    ap.add_argument("--out-meta", default="./embeddings_meta.json")
    ap.add_argument("--batch", type=int, default=32, help="Batch size for embedding (default: 32 for lower RAM usage)")
    ap.add_argument("--threads", type=int, default=None, help="Number of CPU threads for ONNX runtime (default: all available)")
    ap.add_argument("--query", default="", help="run a semantic-search demo instead of building")
    args = ap.parse_args()

    texts, metas = load_chunks(Path(args.in_file))
    print(f"Loaded {len(texts)} chunks. Initializing model {args.model} (threads={args.threads})...", flush=True)
    model = TextEmbedding(model_name=args.model, threads=args.threads)

    # ---- demo mode: search the existing index by meaning ----
    if args.query:
        emb_path = Path(args.out_emb)
        if not emb_path.exists():
            raise SystemExit(f"{emb_path} not found — run without --query first to build it.")
        matrix = np.load(emb_path)
        # query_embed() applies the model's correct query instruction (e.g. BGE prefix).
        qvec = l2_normalize(np.array(list(model.query_embed(args.query))[0]))
        scores = matrix @ qvec
        top = np.argsort(-scores)[:5]
        print(f'\nTop 5 for: "{args.query}"\n' + "-" * 60, flush=True)
        for rank, i in enumerate(top, 1):
            m = metas[i]
            print(f"{rank}. score={scores[i]:.3f} | {m['source']} "
                  f"| audience={m['audience']} | topic={m['topic']}")
            print(f"   section: {m.get('section','')}")
            print(f"   {texts[i][:220].replace(chr(10),' ')}\n", flush=True)
        return

    # ---- build mode: embed every chunk with memory-safe preallocated buffer ----
    total = len(texts)
    print(f"Embedding {total} chunks in batches of {args.batch}...", flush=True)

    # Pre-allocate numpy array to avoid Python list overhead and memory fragmentation
    embeddings_matrix = None

    for i, vec in enumerate(model.embed(texts, batch_size=args.batch)):
        if embeddings_matrix is None:
            dim = len(vec)
            embeddings_matrix = np.empty((total, dim), dtype=np.float32)

        embeddings_matrix[i] = vec
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"  embedded {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)", flush=True)

    if embeddings_matrix is None:
        raise SystemExit("No chunks found to embed.")

    print("Normalizing embeddings...", flush=True)
    embeddings = l2_normalize(embeddings_matrix)
    del embeddings_matrix  # free preallocated buffer

    np.save(args.out_emb, embeddings)
    Path(args.out_meta).write_text(json.dumps({
        "model": args.model,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "normalized": True,
        "source": str(args.in_file),
        "note": "row i aligns with line i of the source jsonl",
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {embeddings.shape[0]} vectors of dim {embeddings.shape[1]} "
          f"-> {args.out_emb}  ({embeddings.nbytes/1024/1024:.1f} MB)", flush=True)
    print(f'Try:  python embed_chunks.py --query "how much is the M.Tech fee?"', flush=True)


if __name__ == "__main__":
    main()
