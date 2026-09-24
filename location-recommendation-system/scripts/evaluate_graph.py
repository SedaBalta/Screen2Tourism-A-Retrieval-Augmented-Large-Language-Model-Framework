import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.graph_recommender import SimilarLocationIndex  # noqa: E402

SETTINGS = [("Degree-normalised co-occurrence", 0.0, 0.0),
            ("+ genre agreement", 0.3, 0.0),
            ("+ province match", 0.0, 0.1),
            ("+ both", 0.3, 0.1)]


def evaluate(index, pairs, w_genre, w_province, ks=(1, 3, 5, 10), min_locations=3):
    by_production = collections.defaultdict(set)
    for _, r in pairs.iterrows():
        by_production[r["yapim_id"]].add(int(r["lokasyon_id"]))
    usable = {p: sorted(l) for p, l in by_production.items() if len(l) >= min_locations}

    hits = {k: [] for k in ks}
    reciprocal = []
    for locs in usable.values():
        for held_out in locs:
            seeds = [l for l in locs if l != held_out]
            ranked = [l for l, _ in index.recommend(seeds, k=max(ks), w_genre=w_genre,
                                                    w_province=w_province, exclude=seeds)]
            for k in ks:
                hits[k].append(1.0 if held_out in ranked[:k] else 0.0)
            reciprocal.append(1.0 / (ranked.index(held_out) + 1) if held_out in ranked else 0.0)
    return ({k: float(np.mean(v) * 100) for k, v in hits.items()},
            float(np.mean(reciprocal)), len(reciprocal), len(usable))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=str(ROOT / "data/catalogue"))
    ap.add_argument("--demo", type=int, default=None, help="print the ten locations most related to this id")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cat = Path(a.catalogue)
    pairs = pd.read_csv(cat / "yapim_lokasyon.csv", encoding="utf-8-sig")
    productions = pd.read_csv(cat / "yapimlar.csv", encoding="utf-8-sig")
    locations = pd.read_csv(cat / "lokasyonlar.csv", encoding="utf-8-sig")
    index = SimilarLocationIndex(pairs, productions, locations)
    print(f"graph: {len(index.degree)} locations, {index.n_edges} undirected edges")

    if a.demo is not None:
        names = dict(zip(locations["id"].astype(int), locations["lokasyon_adi"].astype(str)))
        print(f"\nrelated to {a.demo} ({names.get(a.demo, '?')}):")
        for loc, score in index.recommend(a.demo, k=10):
            print(f"  {score:7.4f}  {loc:>4}  {names.get(loc, '?')}")
        return

    results = {}
    print(f"\n{'method':<36}{'HR@1':>8}{'HR@5':>8}{'HR@10':>8}{'MRR':>8}")
    for name, w_genre, w_province in SETTINGS:
        hr, mrr, n_queries, n_productions = evaluate(index, pairs, w_genre, w_province)
        results[name] = {"hit_rate": hr, "mrr": mrr}
        print(f"{name:<36}{hr[1]:>7.2f}%{hr[5]:>7.2f}%{hr[10]:>7.2f}%{mrr:>8.4f}")
    print(f"\n{n_queries} queries from {n_productions} productions; {len(index.degree)} candidate locations")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n_locations": len(index.degree), "n_edges": index.n_edges,
                   "n_queries": n_queries, "n_productions": n_productions, "results": results},
                  open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
