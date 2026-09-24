import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def weighted_kappa(a, b, k=3):
    a, b = np.asarray(a), np.asarray(b)
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[x, y] += 1
    weights = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    expected = np.outer(np.bincount(a, minlength=k), np.bincount(b, minlength=k)) / len(a)
    denominator = (weights * expected).sum()
    return 1 - (weights * observed).sum() / denominator if denominator > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default=str(ROOT / "data/annotation/human_subset.xlsx"))
    ap.add_argument("--llm", default=str(ROOT / "data/annotation/annotation_llm.xlsx"))
    ap.add_argument("--scores", default=None,
                    help="comma-separated expert scores in file row order, or a path to a file with one score per line; "
                         "written into the human_score column before comparison")
    a = ap.parse_args()

    human = pd.read_excel(a.human)
    if a.scores:
        raw = Path(a.scores).read_text(encoding="utf-8") if Path(a.scores).exists() else a.scores
        scores = [int(x) for x in raw.replace("\n", ",").split(",") if x.strip()]
        if len(scores) != len(human):
            sys.exit(f"{len(scores)} scores for {len(human)} rows")
        human["human_score"] = scores
        human.to_excel(a.human, index=False)

    human["human_score"] = pd.to_numeric(human["human_score"], errors="coerce")
    human = human.dropna(subset=["human_score"])
    llm = pd.read_excel(a.llm)
    merged = human.merge(llm[["query_id", "candidate_id", "llm_score"]],
                         on=["query_id", "candidate_id"], how="inner")
    h = merged["human_score"].astype(int).values
    l = merged["llm_score"].astype(int).values
    print(f"paired assessments: {len(merged)}\n")

    print("confusion matrix (rows expert, columns LLM)")
    print(pd.crosstab(pd.Series(h, name="expert"), pd.Series(l, name="llm"))
          .reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).to_string())
    print(f"\nexact agreement        : {(h == l).mean() * 100:5.1f}%")
    print(f"within one point       : {(abs(h - l) <= 1).mean() * 100:5.1f}%")
    print(f"quadratic-weighted kappa: {weighted_kappa(h, l):.3f}")

    from scipy.stats import spearmanr
    rho, p = spearmanr(h, l)
    print(f"Spearman rho           : {rho:.3f} (p={p:.4g})")


if __name__ == "__main__":
    main()
