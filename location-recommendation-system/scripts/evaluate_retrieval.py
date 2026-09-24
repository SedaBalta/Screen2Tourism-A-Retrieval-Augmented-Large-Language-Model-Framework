import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.text_utils import turkish_lower  # noqa: E402


def ranking_metrics(orders, golds):
    ranks = np.array([o.index(g) for o, g in zip(orders, golds)])
    out = {f"hr{k}": float((ranks < k).mean() * 100) for k in (1, 3, 5, 10)}
    out["p5"] = float((ranks < 5).mean() / 5 * 100)
    out["mrr"] = float(np.mean(1.0 / (ranks + 1)))
    out["ndcg5"] = float(np.mean([1.0 / np.log2(r + 2) if r < 5 else 0.0 for r in ranks]))
    return out


def dense_orders(Q, P):
    return [np.argsort(Q[i] @ P.T)[::-1].tolist() for i in range(len(Q))]


def rerank(model, queries, passages, orders, weight, top_n):
    new_orders, elapsed = [], 0.0
    for qi, q in enumerate(queries):
        candidates = orders[qi][:top_n]
        t0 = time.perf_counter()
        cross = np.asarray(model.predict([[q, passages[j]] for j in candidates], show_progress_bar=False), dtype=float)
        elapsed += time.perf_counter() - t0
        prior = np.linspace(1.0, 0.0, len(candidates))
        span = cross.max() - cross.min()
        cross = (cross - cross.min()) / span if span > 0 else np.full_like(cross, 0.5)
        combined = (1 - weight) * prior + weight * cross
        reordered = [candidates[i] for i in np.argsort(combined)[::-1]]
        new_orders.append(reordered + [j for j in orders[qi] if j not in set(candidates)])
    return new_orders, elapsed / len(queries) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/retrieval/sbert_test.json"))
    ap.add_argument("--encoder", default=str(ROOT / "models/retriever_e5_small"))
    ap.add_argument("--ner", default=str(ROOT / "models/ner_bert_turkish"))
    ap.add_argument("--base-model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--locations", default=str(ROOT / "data/catalogue/lokasyonlar.csv"))
    ap.add_argument("--rerankers", nargs="*", default=[],
                    help="cross-encoder model names to evaluate as second-stage rankers")
    ap.add_argument("--rerank-weights", nargs="*", type=float, default=[0.5])
    ap.add_argument("--rerank-top-n", type=int, default=20)
    ap.add_argument("--rerank-all-arms", action="store_true",
                    help="also rerank the arms without entity filtering and without fine-tuning")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from sentence_transformers import CrossEncoder, SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from recommender.entity_extraction import EntityExtractor

    test = json.load(open(a.data, encoding="utf-8"))
    passages = sorted(set(r["passage"] for r in test))
    index = {p: i for i, p in enumerate(passages)}
    golds = [index[r["passage"]] for r in test]
    raw_queries = [r["query"] for r in test]
    print(f"{len(test)} queries | {len(passages)} candidate passages\n")

    extractor = EntityExtractor(a.ner)
    filtered_queries = [extractor.extract(q) or q for q in raw_queries]

    encoder = SentenceTransformer(a.encoder)
    base = SentenceTransformer(a.base_model)
    enc = lambda m, texts: m.encode(texts, batch_size=32, convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False)

    t0 = time.perf_counter()
    P_ft = enc(encoder, passages)
    index_build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    Q_filtered = enc(encoder, filtered_queries)
    encode_ms = (time.perf_counter() - t0) / len(test) * 1000
    Q_raw = enc(encoder, raw_queries)
    P_base = enc(base, [f"passage: {p}" for p in passages])
    Q_base = enc(base, [f"query: {q}" for q in filtered_queries])

    t0 = time.perf_counter()
    deployed = dense_orders(Q_filtered, P_ft)
    score_ms = (time.perf_counter() - t0) / len(test) * 1000

    arms = {
        "Fine-tuned encoder + entity filtering": deployed,
        "Without entity filtering": dense_orders(Q_raw, P_ft),
        "Without task-specific training (pretrained E5)": dense_orders(Q_base, P_base),
    }
    arm_queries = {
        "Fine-tuned encoder + entity filtering": filtered_queries,
        "Without entity filtering": raw_queries,
        "Without task-specific training (pretrained E5)": filtered_queries,
    }

    vec = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)
    Pt = vec.fit_transform([turkish_lower(p) for p in passages])
    Qt = vec.transform([turkish_lower(q) for q in raw_queries])
    arms["TF-IDF"] = [np.argsort((Qt[i] @ Pt.T).toarray().ravel())[::-1].tolist() for i in range(len(test))]
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([re.findall(r"\w+", turkish_lower(p)) for p in passages])
        arms["BM25"] = [np.argsort(bm25.get_scores(re.findall(r"\w+", turkish_lower(q))))[::-1].tolist()
                        for q in raw_queries]
    except ImportError:
        print("rank_bm25 not installed; BM25 skipped\n")

    results = {name: ranking_metrics(orders, golds) for name, orders in arms.items()}
    latency = {"index_build_s": index_build_s, "encode_ms": encode_ms, "score_ms": score_ms,
               "total_ms": encode_ms + score_ms}

    for model_name in a.rerankers:
        model = CrossEncoder(model_name)
        targets = list(arm_queries) if a.rerank_all_arms else list(arm_queries)[:1]
        for arm_name in targets:
            for w in a.rerank_weights:
                reranked, ms = rerank(model, arm_queries[arm_name], passages, arms[arm_name], w, a.rerank_top_n)
                key = f"{arm_name} + {model_name.split('/')[-1]} (w={w})"
                results[key] = ranking_metrics(reranked, golds)
                results[key]["rerank_ms"] = ms

    print(f"{'setting':<72}{'HR@1':>8}{'HR@5':>8}{'HR@10':>8}{'MRR':>8}{'nDCG@5':>8}")
    for name, r in results.items():
        print(f"{name:<72}{r['hr1']:>7.2f}%{r['hr5']:>7.2f}%{r['hr10']:>7.2f}%{r['mrr']:>8.4f}{r['ndcg5']:>8.4f}")

    print(f"\nlatency: encode {encode_ms:.2f} ms + score {score_ms:.3f} ms = {encode_ms + score_ms:.2f} ms per query"
          f"  (index build {index_build_s:.2f} s)")
    for name, r in results.items():
        if "rerank_ms" in r:
            print(f"  reranking {name}: {r['rerank_ms']:.1f} ms per query")

    locations = pd.read_csv(a.locations, encoding="utf-8-sig")
    names = dict(zip(locations["id"].astype(int), locations["lokasyon_adi"].astype(str)))
    named = np.array([turkish_lower(names.get(int(r["lokasyon_id"]), "")) in turkish_lower(r["query"])
                      for r in test])
    ranks = np.array([o.index(g) for o, g in zip(deployed, golds)])
    subsets = {}
    for label, mask in (("names the location", named), ("does not name the location", ~named)):
        subsets[label] = {"n": int(mask.sum()), "hr1": float((ranks[mask] < 1).mean() * 100),
                          "hr5": float((ranks[mask] < 5).mean() * 100)}
    print("\nqueries by presence of the location name (fine-tuned encoder + entity filtering)")
    for label, s in subsets.items():
        print(f"  {label:<28} n={s['n']:<4} HR@1={s['hr1']:6.2f}%  HR@5={s['hr5']:6.2f}%")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n_queries": len(test), "n_passages": len(passages), "results": results,
                   "latency": latency, "location_name_subsets": subsets},
                  open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
