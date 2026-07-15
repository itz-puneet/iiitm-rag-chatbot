"""
Stage 5: load the embeddings + metadata into a local FAISS vector database.

FAISS indexes vectors only, so a vector store is really two aligned parts:
    faiss_store/index.faiss    the FAISS index (similarity search)
    faiss_store/docstore.jsonl the text + metadata for each vector
    faiss_store/config.json    embedding model name (so queries use the same one)
Row i of the index  <->  line i of docstore.jsonl  <->  line i of chunks.jsonl.

The vectors from embed_chunks.py are L2-normalized, so we use IndexFlatIP
(inner product == cosine similarity). At 13k vectors an exact flat index is
instant; no approximate index (IVF/HNSW) is needed.

Metadata filtering (audience / topic) is done with FAISS's native IDSelector:
we compute the matching row-ids first, then restrict the search to them — so
filtering happens inside FAISS, not as a wasteful post-filter.

Setup:
    pip install faiss-cpu fastembed

Usage:
    python build_faiss.py                                   # build faiss_store/
    python build_faiss.py --query "M.Tech hostel fees"      # search
    python build_faiss.py --query "credit requirements" --audience MTech --topic Curriculum
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np


def load_docstore(chunks_path: Path):
    docs = []
    with open(chunks_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))       # {"text": ..., "metadata": {...}}
    return docs


def build(args) -> None:
    emb = np.load(args.embeddings).astype("float32")
    docs = load_docstore(Path(args.chunks))
    if len(docs) != emb.shape[0]:
        raise SystemExit(f"Mismatch: {emb.shape[0]} vectors vs {len(docs)} chunks — "
                         "they must be aligned (rebuild embeddings from the same chunks.jsonl).")

    index = faiss.IndexFlatIP(emb.shape[1])          # cosine sim on normalized vectors
    index.add(emb)

    out = Path(args.store)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "index.faiss"))
    with open(out / "docstore.jsonl", "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    model = "unknown"
    meta_path = Path(args.embeddings).with_name("embeddings_meta.json")
    if meta_path.exists():
        model = json.loads(meta_path.read_text(encoding="utf-8")).get("model", "unknown")
    (out / "config.json").write_text(
        json.dumps({"model": model, "dim": int(emb.shape[1]), "count": index.ntotal,
                    "metric": "inner_product", "normalized": True}, indent=2),
        encoding="utf-8")

    print(f"FAISS store built at {out.resolve()}")
    print(f"  index.faiss    : {index.ntotal} vectors, dim {emb.shape[1]}")
    print(f"  docstore.jsonl : {len(docs)} records")
    print(f"  model          : {model}")


def matches(meta: dict, audience: str, topic: str) -> bool:
    """Retrieval semantics: a program query also matches institute-wide ('General')
    chunks; audience is a LIST so we test membership, not equality."""
    if audience and not (audience in meta["audience"] or "General" in meta["audience"]):
        return False
    if topic and meta.get("topic") != topic:
        return False
    return True


def search(args) -> None:
    from fastembed import TextEmbedding

    out = Path(args.store)
    index = faiss.read_index(str(out / "index.faiss"))
    docs = load_docstore(out / "docstore.jsonl")
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))

    # Embed the query with the SAME model (query_embed applies the BGE prefix).
    model = TextEmbedding(model_name=cfg["model"])
    qv = np.array(list(model.query_embed(args.query))[0], dtype="float32")
    qv /= np.linalg.norm(qv) + 1e-12
    qv = qv.reshape(1, -1)

    # Pre-filter: restrict the search to row-ids whose metadata matches.
    params = None
    if args.audience or args.topic:
        ids = np.array([i for i, d in enumerate(docs)
                        if matches(d["metadata"], args.audience, args.topic)], dtype="int64")
        if ids.size == 0:
            print("No chunks match that audience/topic filter."); return
        params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids))
        print(f"(filtering to {ids.size} chunks: "
              f"audience={args.audience or 'any'}, topic={args.topic or 'any'})")

    scores, idx = index.search(qv, args.k, params=params)

    print(f'\nTop {args.k} for: "{args.query}"\n' + "-" * 60)
    for rank, (i, s) in enumerate(zip(idx[0], scores[0]), 1):
        if i == -1:
            continue
        m = docs[i]["metadata"]
        print(f"{rank}. score={s:.3f} | {m['source']} | audience={m['audience']} | topic={m['topic']}")
        print(f"   section: {m.get('section','')}")
        print(f"   {docs[i]['text'][:220].replace(chr(10),' ')}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/query a local FAISS vector store.")
    ap.add_argument("--embeddings", default="./embeddings.npy")
    ap.add_argument("--chunks", default="./chunks.jsonl")
    ap.add_argument("--store", default="./faiss_store")
    ap.add_argument("--query", default="", help="search instead of build")
    ap.add_argument("--audience", default="", help="filter: BTech/MTech/MBA/MS/PhD/IPG")
    ap.add_argument("--topic", default="", help="filter: Fees/Curriculum/Hostel/...")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    search(args) if args.query else build(args)


if __name__ == "__main__":
    main()
