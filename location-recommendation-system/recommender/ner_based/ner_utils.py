import json

import torch
from torch.utils.data import Dataset


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_labels(path):
    cfg = json.load(open(path, encoding="utf-8"))
    return cfg["labels"], cfg["label2id"], {int(k): v for k, v in cfg["id2label"].items()}


def align_labels(word_tags, word_ids, label2id):
    aligned, previous = [], None
    for w in word_ids:
        if w is None:
            aligned.append(-100)
        elif w != previous:
            aligned.append(label2id.get(word_tags[w], 0))
        else:
            tag = word_tags[w]
            aligned.append(label2id.get("I-" + tag[2:], 0) if tag.startswith("B-")
                           else label2id.get(tag, 0))
        previous = w
    return aligned


class TokenClassificationDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length, label2id):
        self.items = []
        for r in rows:
            enc = tokenizer(r["tokens"], is_split_into_words=True, truncation=True,
                            padding="max_length", max_length=max_length,
                            return_tensors="pt")
            labels = align_labels(r["ner_tags"], enc.word_ids(batch_index=0), label2id)
            self.items.append({
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def predict_sequences(loader, model, id2label, device):
    model.eval()
    gold, pred = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            predictions = logits.argmax(-1).cpu()
            for p_row, l_row in zip(predictions, batch["labels"]):
                g, h = [], []
                for p, l in zip(p_row.tolist(), l_row.tolist()):
                    if l == -100:
                        continue
                    g.append(id2label[l])
                    h.append(id2label[p])
                gold.append(g)
                pred.append(h)
    return gold, pred


def entity_metrics(gold, pred):
    from seqeval.metrics import (classification_report, f1_score,
                                 precision_score, recall_score)
    report = classification_report(gold, pred, output_dict=True, zero_division=0)
    return {
        "f1": f1_score(gold, pred) * 100,
        "precision": precision_score(gold, pred) * 100,
        "recall": recall_score(gold, pred) * 100,
        "macro_f1": report["macro avg"]["f1-score"] * 100,
        "per_type": {k: v for k, v in report.items()
                     if k not in ("micro avg", "macro avg", "weighted avg")},
    }
