import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.visual_search import load_index, rank_locations  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(ROOT / "data/visual/clip_index.npy"))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    keys, M, locations = load_index(a.index)
    per_location = collections.Counter(locations.tolist())
    evaluable = np.array([per_location[l] >= 2 for l in locations])
    print(f"index: {len(keys)} images over {len(per_location)} locations")
    print(f"queries (locations with at least two images): {int(evaluable.sum())}\n")

    S = M @ M.T
    np.fill_diagonal(S, -np.inf)

    hits = {1: [], 3: [], 5: []}
    precisions, reciprocal = [], []
    for i in np.flatnonzero(evaluable):
        ranked = [loc for loc, _ in rank_locations(S[i], locations, 20)]
        true = int(locations[i])
        for k in hits:
            hits[k].append(1.0 if true in ranked[:k] else 0.0)
        precisions.append(sum(1 for x in ranked[:5] if x == true) / 5)
        reciprocal.append(1.0 / (ranked.index(true) + 1) if true in ranked else 0.0)

    result = {"n_images": len(keys), "n_locations": len(per_location), "n_queries": int(evaluable.sum()),
              **{f"hr{k}": float(np.mean(v) * 100) for k, v in hits.items()},
              "p5": float(np.mean(precisions) * 100), "mrr": float(np.mean(reciprocal))}
    for k in hits:
        print(f"Hit Rate@{k} : {result[f'hr{k}']:6.2f}%")
    print(f"Precision@5: {result['p5']:6.2f}%")
    print(f"MRR        : {result['mrr']:6.4f}")

    if a.bootstrap:
        rng = np.random.RandomState(42)
        arr = np.array(hits[5])
        samples = [arr[rng.randint(0, len(arr), len(arr))].mean() * 100 for _ in range(a.bootstrap)]
        lo, hi = np.percentile(samples, [2.5, 97.5])
        result["hr5_ci95"] = [float(lo), float(hi)]
        print(f"Hit Rate@5 95% CI: [{lo:.2f}%, {hi:.2f}%]")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(a.out, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
