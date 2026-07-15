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
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--query", default="", help="run a semantic-search demo instead of building")
    args = ap.parse_args()

    texts, metas = load_chunks(Path(args.in_file))
    print(f"Loaded {len(texts)} chunks. Loading model {args.model} "
          f"(first run downloads ~130MB from HuggingFace) ...")
    model = TextEmbedding(model_name=args.model)

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
        print(f'\nTop 5 for: "{args.query}"\n' + "-" * 60)
        for rank, i in enumerate(top, 1):
            m = metas[i]
            print(f"{rank}. score={scores[i]:.3f} | {m['source']} "
                  f"| audience={m['audience']} | topic={m['topic']}")
            print(f"   section: {m.get('section','')}")
            print(f"   {texts[i][:220].replace(chr(10),' ')}\n")
        return

    # ---- build mode: embed every chunk ----
    vectors = []
    for i, vec in enumerate(model.embed(texts, batch_size=args.batch), start=1):
        vectors.append(vec)
        if i % 2000 == 0 or i == len(texts):
            print(f"  embedded {i}/{len(texts)}")

    embeddings = l2_normalize(np.array(vectors, dtype="float32"))
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
          f"-> {args.out_emb}  ({embeddings.nbytes/1024/1024:.1f} MB)")
    print(f'Try:  python embed_chunks.py --query "how much is the M.Tech fee?"')


if __name__ == "__main__":
    main()
