import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.ner_utils import (TokenClassificationDataset, entity_metrics,  # noqa: E402
                                   load_jsonl, load_labels, predict_sequences)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/ner_bert_turkish"))
    ap.add_argument("--data", default=str(ROOT / "data/ner"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = Path(a.data)
    _, label2id, id2label = load_labels(data / "ner_labels.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForTokenClassification.from_pretrained(a.model).to(device)

    rows = load_jsonl(data / f"ner_{a.split}.jsonl")
    loader = DataLoader(TokenClassificationDataset(rows, tokenizer, a.max_len, label2id),
                        batch_size=a.batch_size)
    result = entity_metrics(*predict_sequences(loader, model, id2label, device))

    print(f"{a.split}: {len(rows)} sentences")
    print(f"micro  P={result['precision']:.2f}%  R={result['recall']:.2f}%  F1={result['f1']:.2f}%")
    print(f"macro  F1={result['macro_f1']:.2f}%")
    for name, v in result["per_type"].items():
        print(f"  {name:<6} P={v['precision'] * 100:6.2f}%  R={v['recall'] * 100:6.2f}%  "
              f"F1={v['f1-score'] * 100:6.2f}%  n={int(v['support'])}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
