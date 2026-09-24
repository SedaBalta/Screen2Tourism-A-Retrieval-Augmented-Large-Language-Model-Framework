import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]

torch.manual_seed(42)
np.random.seed(42)


def evaluate(model, test_data, k=5, batch_size=32):
    passages = sorted(set(p["passage"] for p in test_data))
    index = {p: i for i, p in enumerate(passages)}
    P = model.encode(passages, batch_size=batch_size, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
    Q = model.encode([p["query"] for p in test_data], batch_size=batch_size,
                     convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    hits, precisions = [], []
    for i, row in enumerate(test_data):
        top = np.argsort(Q[i] @ P.T)[::-1][:k]
        gold = index[row["passage"]]
        hits.append(1.0 if gold in top else 0.0)
        precisions.append(sum(1 for j in top if j == gold) / k)
    return {"hit_rate_at_5": float(np.mean(hits) * 100),
            "precision_at_5": float(np.mean(precisions) * 100),
            "n_queries": len(test_data), "n_passages": len(passages)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--data", default=str(ROOT / "data/retrieval"))
    ap.add_argument("--loss", choices=["mnrl", "cos"], default="mnrl")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--out", default=None, help="results JSON path")
    a = ap.parse_args()

    data = Path(a.data)
    load = lambda name: json.load(open(data / f"sbert_{name}.json", encoding="utf-8"))
    train_data, test_data = load("train"), load("test")

    model = SentenceTransformer(a.base_model)
    if a.loss == "mnrl":
        examples = [InputExample(texts=[p["query"], p["passage"]]) for p in train_data]
        loss = losses.MultipleNegativesRankingLoss(model)
    else:
        examples = [InputExample(texts=[p["query"], p["passage"]], label=1.0) for p in train_data]
        loss = losses.CosineSimilarityLoss(model)
    loader = DataLoader(examples, shuffle=True, batch_size=a.batch_size)

    model.fit(train_objectives=[(loader, loss)], epochs=a.epochs,
              warmup_steps=int(len(loader) * a.epochs * 0.1), show_progress_bar=False)

    result = evaluate(model, test_data)
    print(f"Hit Rate@5 = {result['hit_rate_at_5']:.2f}%   Precision@5 = {result['precision_at_5']:.2f}%  "
          f"({result['n_queries']} queries, {result['n_passages']} passages)")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"base_model": a.base_model, "loss": a.loss, "epochs": a.epochs,
                   "batch_size": a.batch_size, **result},
                  open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    if a.save_dir:
        model.save(a.save_dir)
        print(f"checkpoint written to {a.save_dir}")


if __name__ == "__main__":
    main()
