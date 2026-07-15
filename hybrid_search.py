"""
Stage 6: Hybrid search = semantic (FAISS) + keyword (BM25), fused with RRF.

Why hybrid: vector search finds meaning ("cost to stay on campus" -> hostel fee)
but blurs exact identifiers ("Rule 14.2" ~ "Rule 14.3"). BM25 nails exact tokens
(rule/form/clause numbers) but misses paraphrases. Reciprocal Rank Fusion (RRF)
merges the two RANKINGS, so it needs no score normalization between the
incompatible scales (cosine similarity vs unbounded BM25 scores).

Reuses the FAISS store from build_faiss.py (faiss_store/) and the same
open-source embedding model. Everything is free and local.

Setup:
    pip install rank-bm25 faiss-cpu fastembed

Usage:
    python hybrid_search.py --query "thesis credit requirement R.7"
    python hybrid_search.py --query "how much to live on campus" --mode vector
    python hybrid_search.py --query "Rule 4.1.2 registration" --audience MTech --topic Ordinance/Rules
    #   --mode hybrid|vector|bm25   (compare retrievers)
"""

import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

# Tokenizer that KEEPS dotted identifiers together: "Rule 14.2" -> ["rule","14.2"],
# "R.7" -> ["r.7"], "Form B" -> ["form","b"]. A naive split would shatter "14.2"
# into "14" and "2" and lose the exact match that keyword search exists to catch.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def tokenize(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())


def matches(meta: dict, audience: str, topic: str) -> bool:
    """Same retrieval semantics as the vector store: a program query also matches
    institute-wide ('General') chunks; audience is a LIST (membership, not =)."""
    if audience and not (audience in meta["audience"] or "General" in meta["audience"]):
        return False
    if topic and meta.get("topic") != topic:
        return False
    return True


class HybridRetriever:
    def __init__(self, store: str = "./faiss_store"):
        from fastembed import TextEmbedding

        store = Path(store)
        self.index = faiss.read_index(str(store / "index.faiss"))
        self.docs = [json.loads(l) for l in open(store / "docstore.jsonl", encoding="utf-8") if l.strip()]
        cfg = json.loads((store / "config.json").read_text(encoding="utf-8"))
        self.model = TextEmbedding(model_name=cfg["model"])
        # BM25 over the same chunk texts (built in-memory; ~1s for 13k chunks).
        self.bm25 = BM25Okapi([tokenize(d["text"]) for d in self.docs])
        self._reranker = None          # cross-encoder, lazy-loaded on first rerank
        self._reranker_name = None

    def _embed_query(self, query: str) -> np.ndarray:
        vec = np.array(list(self.model.query_embed(query))[0], dtype="float32")
        vec /= np.linalg.norm(vec) + 1e-12
        return vec.reshape(1, -1)

    def _allowed_ids(self, audience: str, topic: str):
        if not (audience or topic):
            return None                      # no filter -> whole corpus
        return np.array([i for i, d in enumerate(self.docs)
                         if matches(d["metadata"], audience, topic)], dtype="int64")

    def _vector_ranking(self, query, allowed, candidates):
        params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed)) if allowed is not None else None
        scores, idx = self.index.search(self._embed_query(query), candidates, params=params)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]

    def _bm25_ranking(self, query, allowed, candidates):
        scores = self.bm25.get_scores(tokenize(query))
        allowed_set = set(allowed.tolist()) if allowed is not None else None
        ranked = sorted(
            (i for i in range(len(scores))
             if scores[i] > 0 and (allowed_set is None or i in allowed_set)),
            key=lambda i: scores[i], reverse=True,
        )[:candidates]
        return [(i, float(scores[i])) for i in ranked]

    def _rerank(self, query, ids, model_name):
        """Cross-encoder rerank: score each (query, chunk) pair directly and sort.
        Uses fastembed's ONNX TextCrossEncoder (no torch). A cross-encoder reads
        query and chunk together, so it judges true relevance far better than the
        bi-encoder + BM25 signals that fusion combines — its job here is to demote
        confident-but-irrelevant hits (e.g. vector noise on a bare-number query)."""
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        if self._reranker is None or self._reranker_name != model_name:
            self._reranker = TextCrossEncoder(model_name=model_name)
            self._reranker_name = model_name
        scores = list(self._reranker.rerank(query, [self.docs[i]["text"] for i in ids]))
        return sorted(zip(ids, scores), key=lambda t: t[1], reverse=True)

    def search(self, query, k=5, audience="", topic="", mode="hybrid",
               candidates=50, k_rrf=15, rerank=False, rerank_pool=20,
               rerank_model="Xenova/ms-marco-MiniLM-L-6-v2"):
        """Return top-k results. mode: 'hybrid' (RRF), 'vector', or 'bm25'.

        k_rrf=15 (not the web-search-default 60) on purpose: our two retrievers
        are authoritative for DISJOINT query types (BM25 for exact IDs like
        'Rule 30.7', vectors for paraphrases), so their lists frequently do not
        overlap. A large k flattens rank differences and lets a chunk sitting
        mid-list in BOTH rankers outrank the exact-match chunk that is #1 in one
        ranker but absent from the other — burying exactly what we built BM25 to
        catch. A smaller k keeps each retriever's strong top hits near the top.

        rerank=True: fuse a larger pool (rerank_pool), then reorder it with a
        cross-encoder and return the top k. This is the accuracy-maximizing setup.
        """
        allowed = self._allowed_ids(audience, topic)
        vec = self._vector_ranking(query, allowed, candidates) if mode in ("hybrid", "vector") else []
        bm = self._bm25_ranking(query, allowed, candidates) if mode in ("hybrid", "bm25") else []

        vrank = {i: r for r, (i, _) in enumerate(vec, 1)}
        brank = {i: r for r, (i, _) in enumerate(bm, 1)}
        vscore = dict(vec)
        bscore = dict(bm)

        # Reciprocal Rank Fusion: sum 1/(k_rrf + rank) across the rankers a doc appears in.
        fused = {}
        for i in set(vrank) | set(brank):
            score = 0.0
            if i in vrank:
                score += 1.0 / (k_rrf + vrank[i])
            if i in brank:
                score += 1.0 / (k_rrf + brank[i])
            fused[i] = score

        order = sorted(fused, key=lambda i: fused[i], reverse=True)

        # Optional cross-encoder rerank of the fused top pool. Rerank at least k
        # candidates (max(rerank_pool, k)) so a large k is never silently capped
        # to rerank_pool; the un-reranked tail is appended to backfill if needed.
        rerank_score = {}
        if rerank and order:
            pool = order[:max(rerank_pool, k)]
            for i, s in self._rerank(query, pool, rerank_model):
                rerank_score[i] = float(s)
            reranked = sorted(rerank_score, key=lambda i: rerank_score[i], reverse=True)
            order = reranked + [i for i in order if i not in rerank_score]

        results = []
        for i in order[:k]:
            results.append({
                "id": i,
                "text": self.docs[i]["text"],
                "metadata": self.docs[i]["metadata"],
                "rrf": round(fused[i], 5),
                "rerank_score": round(rerank_score[i], 3) if i in rerank_score else None,
                "vec_rank": vrank.get(i), "bm25_rank": brank.get(i),
                "vec_score": round(vscore[i], 3) if i in vscore else None,
                "bm25_score": round(bscore[i], 2) if i in bscore else None,
            })
        return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid (vector + BM25) search over the FAISS store.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--store", default="./faiss_store")
    ap.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    ap.add_argument("--audience", default="")
    ap.add_argument("--topic", default="")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rerank", action="store_true", help="cross-encoder rerank the fused pool")
    args = ap.parse_args()

    r = HybridRetriever(args.store)
    hits = r.search(args.query, k=args.k, audience=args.audience, topic=args.topic,
                    mode=args.mode, rerank=args.rerank)

    flt = f" | audience={args.audience or 'any'} topic={args.topic or 'any'}"
    print(f'\n[{args.mode}] "{args.query}"{flt}\n' + "-" * 68)
    if not hits:
        print("(no results)")
        return
    for rank, h in enumerate(hits, 1):
        m = h["metadata"]
        rr = f"rerank={h['rerank_score']} | " if h.get("rerank_score") is not None else ""
        prov = f"{rr}rrf={h['rrf']} (vec#{h['vec_rank']} score={h['vec_score']}, bm25#{h['bm25_rank']} score={h['bm25_score']})"
        print(f"{rank}. {m['source']} | audience={m['audience']} | topic={m['topic']}")
        print(f"   {prov}")
        print(f"   section: {m.get('section','')}")
        print(f"   {h['text'][:200].replace(chr(10),' ')}\n")


if __name__ == "__main__":
    main()
