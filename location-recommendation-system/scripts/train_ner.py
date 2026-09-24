import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.ner_utils import (TokenClassificationDataset, entity_metrics,  # noqa: E402
                                   load_jsonl, load_labels, predict_sequences)

torch.manual_seed(42)
np.random.seed(42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dbmdz/bert-base-turkish-cased")
    ap.add_argument("--data", default=str(ROOT / "data/ner"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--save-dir", default=None, help="write the fine-tuned checkpoint here")
    ap.add_argument("--out", default=None, help="results JSON path")
    a = ap.parse_args()

    data = Path(a.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, label2id, id2label = load_labels(data / "ner_labels.json")

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    make = lambda name: TokenClassificationDataset(
        load_jsonl(data / f"ner_{name}.jsonl"), tokenizer, a.max_len, label2id)
    train_loader = DataLoader(make("train"), batch_size=a.batch_size, shuffle=True)
    dev_loader = DataLoader(make("dev"), batch_size=a.batch_size)
    test_loader = DataLoader(make("test"), batch_size=a.batch_size)

    model = AutoModelForTokenClassification.from_pretrained(
        a.model, num_labels=len(labels), id2label=id2label, label2id=label2id).to(device)
    optimizer = AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total_steps = len(train_loader) * a.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    for epoch in range(1, a.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
        dev = entity_metrics(*predict_sequences(dev_loader, model, id2label, device))
        print(f"epoch {epoch}/{a.epochs}  loss={total_loss / len(train_loader):.4f}  dev F1={dev['f1']:.2f}%")

    test = entity_metrics(*predict_sequences(test_loader, model, id2label, device))
    print(f"\ntest  F1={test['f1']:.2f}%  P={test['precision']:.2f}%  R={test['recall']:.2f}%  "
          f"macro F1={test['macro_f1']:.2f}%")
    for name, v in test["per_type"].items():
        print(f"  {name:<6} P={v['precision'] * 100:6.2f}%  R={v['recall'] * 100:6.2f}%  "
              f"F1={v['f1-score'] * 100:6.2f}%  n={int(v['support'])}")

    if a.out:
        result = {"model": a.model, "epochs": a.epochs, "lr": a.lr,
                  "batch_size": a.batch_size, "max_len": a.max_len, **test}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False, default=float)
    if a.save_dir:
        model.save_pretrained(a.save_dir)
        tokenizer.save_pretrained(a.save_dir)
        print(f"checkpoint written to {a.save_dir}")


if __name__ == "__main__":
    main()
