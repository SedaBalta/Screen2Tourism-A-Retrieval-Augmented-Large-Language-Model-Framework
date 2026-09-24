import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def dcg(grades):
    return sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(grades))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "data/annotation/annotation_llm.xlsx"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = pd.read_excel(a.file) if a.file.endswith(".xlsx") else pd.read_csv(a.file, encoding="utf-8-sig")
    for c in ("annotator_1", "annotator_2"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["annotator_1", "annotator_2"]).copy()
    if len(df) == 0:
        sys.exit("no completed annotations")
    print(f"annotated pairs: {len(df)}  queries: {df['query_id'].nunique()}")

    df["grade"] = df[["annotator_1", "annotator_2"]].mean(axis=1)
    df["rel_strict"] = (df["grade"] >= 1.5).astype(int)
    df["rel_lenient"] = (df["grade"] >= 0.5).astype(int)

    result = {"n_pairs": int(len(df)), "n_queries": int(df["query_id"].nunique())}
    groups = [g.sort_values("_system_rank") for _, g in df.groupby("query_id")]

    for mode in ("strict", "lenient"):
        col = f"rel_{mode}"
        p_at_k = [g.head(a.k)[col].mean() for g in groups]
        hit = [1.0 if g.head(a.k)[col].sum() > 0 else 0.0 for g in groups]
        result[mode] = {"p_at_k": float(np.mean(p_at_k) * 100), "hit_rate": float(np.mean(hit) * 100)}
        print(f"\n{mode}: Precision@{a.k}={result[mode]['p_at_k']:.2f}%  Hit Rate@{a.k}={result[mode]['hit_rate']:.2f}%")

    ndcg = []
    for g in groups:
        ideal = dcg(sorted(g["grade"], reverse=True)[:a.k])
        ndcg.append(dcg(g.head(a.k)["grade"].tolist()) / ideal if ideal > 0 else 0.0)
    result["ndcg"] = float(np.mean(ndcg))
    print(f"\nnDCG@{a.k} (ordinal grades) = {result['ndcg']:.4f}")

    single_p = [g.head(a.k)["_is_gold"].mean() for g in groups]
    single_hit = [1.0 if g.head(a.k)["_is_gold"].sum() > 0 else 0.0 for g in groups]
    result["single_target"] = {"p_at_k": float(np.mean(single_p) * 100), "hit_rate": float(np.mean(single_hit) * 100)}
    print(f"historical site only: Precision@{a.k}={result['single_target']['p_at_k']:.2f}%  "
          f"Hit Rate@{a.k}={result['single_target']['hit_rate']:.2f}%")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(a.out, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
