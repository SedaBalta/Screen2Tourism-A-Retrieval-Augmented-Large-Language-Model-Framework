import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.genre_classifier import GenreClassifier, build_pipeline  # noqa: E402

torch.manual_seed(42)
np.random.seed(42)

SETTINGS = ["logreg", "logreg_unweighted", "svc", "rf", "mlp", "bert_5e-5", "bert_2e-5"]


def metrics(y_true, scores):
    pred = scores.argmax(1)
    order = np.argsort(scores, axis=1)[:, ::-1]
    return {"top1": accuracy_score(y_true, pred) * 100,
            "top3": float(np.mean([y_true[i] in order[i, :3] for i in range(len(y_true))]) * 100),
            "macro_f1": f1_score(y_true, pred, average="macro") * 100}


def tfidf_features(train_texts, test_texts):
    vec = TfidfVectorizer(ngram_range=(1, 3), max_features=15000, sublinear_tf=True, min_df=2)
    return vec.fit_transform(train_texts), vec.transform(test_texts)


def run_logreg(train, test, class_weight):
    clf = build_pipeline(class_weight=class_weight)
    clf.fit(train["senaryo_ozeti"].fillna(""), train["label_id"].values)
    return clf.predict_proba(test["senaryo_ozeti"].fillna("")), clf


def run_svc(train, test):
    X, Xt = tfidf_features(train["senaryo_ozeti"].fillna(""), test["senaryo_ozeti"].fillna(""))
    clf = LinearSVC(C=1.0, class_weight="balanced", random_state=42).fit(X, train["label_id"].values)
    return clf.decision_function(Xt)


def run_rf(train, test):
    X, Xt = tfidf_features(train["senaryo_ozeti"].fillna(""), test["senaryo_ozeti"].fillna(""))
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                 random_state=42).fit(X, train["label_id"].values)
    return clf.predict_proba(Xt)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3))
        self.res = ResidualBlock(256)
        self.head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.head(self.res(self.proj(x)))


def run_mlp(train, test, device, epochs=25):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("intfloat/multilingual-e5-small")
    encode = lambda texts, prefix: encoder.encode([f"{prefix}: {t}" for t in texts], batch_size=32,
                                                  show_progress_bar=False, convert_to_numpy=True)
    Xtr = encode(train["senaryo_ozeti"].fillna("").tolist(), "passage")
    Xte = encode(test["senaryo_ozeti"].fillna("").tolist(), "query")
    ytr = train["label_id"].values
    n_classes = int(max(ytr.max(), test["label_id"].max())) + 1
    counts = np.bincount(ytr, minlength=n_classes)
    weights = torch.tensor(len(ytr) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                      torch.tensor(ytr, dtype=torch.long)), batch_size=32, shuffle=True)
    model = ResidualMLP(Xtr.shape[1], n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=weights)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb.to(device)), yb.to(device)).backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(Xte, dtype=torch.float32).to(device)), -1).cpu().numpy()


class SummaryDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = df["senaryo_ozeti"].fillna("").tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], truncation=True, padding="max_length",
                             max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "label": torch.tensor(self.labels[i], dtype=torch.long)}


def run_bert(train, test, device, lr, epochs=2, model_name="dbmdz/bert-base-turkish-cased"):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    n_classes = int(train["label_id"].max()) + 1
    train_loader = DataLoader(SummaryDataset(train, tokenizer), batch_size=16, shuffle=True)
    test_loader = DataLoader(SummaryDataset(test, tokenizer), batch_size=16)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total * 0.1), total)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for b in train_loader:
            optimizer.zero_grad()
            logits = model(input_ids=b["input_ids"].to(device),
                           attention_mask=b["attention_mask"].to(device)).logits
            loss = criterion(logits, b["label"].to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"    epoch {epoch}/{epochs}  loss={total_loss / len(train_loader):.4f}")
    model.eval()
    scores = []
    with torch.no_grad():
        for b in test_loader:
            logits = model(input_ids=b["input_ids"].to(device),
                           attention_mask=b["attention_mask"].to(device)).logits
            scores.append(torch.softmax(logits, -1).cpu().numpy())
    return np.vstack(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/genre"))
    ap.add_argument("--settings", nargs="+", default=SETTINGS, choices=SETTINGS)
    ap.add_argument("--train-split", default="train+val", choices=["train+val", "train"],
                    help="fit on train+val (default) or on the train split only")
    ap.add_argument("--save-classifier", default=None,
                    help="write the selected TF-IDF + logistic regression model to this path")
    ap.add_argument("--out", default=None, help="results JSON path")
    a = ap.parse_args()

    data = Path(a.data)
    train = pd.read_csv(data / "genre_train.csv", encoding="utf-8")
    if a.train_split == "train+val":
        train = pd.concat([train, pd.read_csv(data / "genre_val.csv", encoding="utf-8")],
                          ignore_index=True)
    test = pd.read_csv(data / "genre_test.csv", encoding="utf-8")
    label_map = json.load(open(data / "genre_label_map.json", encoding="utf-8"))
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    y_test = test["label_id"].values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{a.train_split}={len(train)}  test={len(test)}  device={device}\n")

    results, per_class = {}, None
    for setting in a.settings:
        print(f"[{setting}]")
        if setting == "logreg":
            scores, clf = run_logreg(train, test, "balanced")
            pred = scores.argmax(1)
            per_class = classification_report(y_test, pred, target_names=[id2label[i] for i in sorted(id2label)],
                                              output_dict=True, zero_division=0)
            if a.save_classifier:
                GenreClassifier(clf, id2label).save(a.save_classifier)
                print(f"    classifier written to {a.save_classifier}")
        elif setting == "logreg_unweighted":
            scores, _ = run_logreg(train, test, None)
        elif setting == "svc":
            scores = run_svc(train, test)
        elif setting == "rf":
            scores = run_rf(train, test)
        elif setting == "mlp":
            scores = run_mlp(train, test, device)
        else:
            scores = run_bert(train, test, device, lr=float(setting.split("_")[1]))
        results[setting] = metrics(y_test, scores)
        r = results[setting]
        print(f"    Top-1={r['top1']:.2f}%  Top-3={r['top3']:.2f}%  Macro F1={r['macro_f1']:.2f}%")

    print(f"\n{'setting':<20}{'Top-1':>9}{'Top-3':>9}{'Macro F1':>10}")
    for name, r in results.items():
        print(f"{name:<20}{r['top1']:>8.2f}%{r['top3']:>8.2f}%{r['macro_f1']:>9.2f}%")

    if per_class:
        print(f"\nper class (logreg)\n{'class':<12}{'P':>9}{'R':>9}{'F1':>9}{'n':>6}")
        for name, v in per_class.items():
            if name in ("accuracy", "macro avg", "weighted avg"):
                continue
            print(f"{name:<12}{v['precision'] * 100:>8.2f}%{v['recall'] * 100:>8.2f}%"
                  f"{v['f1-score'] * 100:>8.2f}%{int(v['support']):>6}")
        m = per_class["macro avg"]
        print(f"{'macro avg':<12}{m['precision'] * 100:>8.2f}%{m['recall'] * 100:>8.2f}%{m['f1-score'] * 100:>8.2f}%")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"settings": results, "per_class_logreg": per_class},
                  open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
