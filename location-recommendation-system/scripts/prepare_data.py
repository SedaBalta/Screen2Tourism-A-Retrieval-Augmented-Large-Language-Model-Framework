import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PLACE_TERMS = [
    "sokak", "cadde", "bulvar", "mahalle", "semt", "meydan", "çarşı", "pazar", "arasta",
    "köy", "kasaba", "şehir", "kent", "ilçe", "sanayi", "bölge", "bölgesi",
    "konak", "köşk", "villa", "yalı", "mansion", "apartman", "bina", "ev", "oda",
    "fabrika", "depo", "hangar", "ambar", "atölye", "tersane",
    "okul", "üniversite", "kampüs", "lise", "ilkokul",
    "hastane", "klinik", "acil", "sağlık",
    "karakol", "emniyet", "adliye", "mahkeme", "cezaevi", "nezarethane",
    "cami", "kilise", "manastır", "türbe", "mescit", "tapınak",
    "sahil", "plaj", "kıyı", "deniz", "göl", "nehir", "ırmak", "dere",
    "orman", "park", "bahçe", "mezarlık", "tarla", "çiftlik", "bağ",
    "dağ", "tepe", "vadi", "kanyon", "yayla", "mağara",
    "gar", "istasyon", "terminal", "köprü", "viyadük", "liman", "iskele", "rıhtım",
    "havalimanı", "havaliman", "havaalanı",
    "kale", "hisar", "burç", "sur", "kervansaray", "han", "hamam", "saray",
    "antik", "arkeolojik", "harabe", "kalıntı", "sütun", "tiyatro",
    "meydanı", "meydanda", "alanı", "semtinde", "sokağında", "caddesinde",
    "gece kulübü", "pavyon", "bar", "kafe", "restoran", "otel", "pansiyon",
]

TIME_TERMS = [
    "sabah", "öğle", "öğleden", "öğleden sonra", "akşam", "gece", "gece yarısı",
    "şafak", "gün doğumu", "gün batımı", "alacakaranlık", "tan",
    "ilkbahar", "bahar", "yaz", "sonbahar", "kış", "kışın", "yazın",
    "yıl", "yılında", "asır", "yüzyıl", "dönem", "devir", "çağ", "tarihinde",
    "ertesi", "önceki", "sonraki", "o gün", "o gece", "o sabah",
    "geçmişte", "eskiden", "bugün", "şimdilerde",
]

EVENT_TERMS = [
    "kovalamaca", "kovalama", "takip", "pusu", "çatışma", "kavga", "dövüş", "saldırı",
    "kaçış", "baskın", "operasyon", "arama", "tarama", "tutuklama", "gözaltı",
    "cinayet", "öldürme", "hırsızlık", "soygun", "kaçakçılık", "uyuşturucu",
    "şantaj", "dolandırıcılık", "sahtecilik",
    "düğün", "nikah", "nişan", "cenaze", "yas", "tören", "kutlama", "parti",
    "kavuşma", "ayrılık", "veda", "buluşma", "randevu", "ilk tanışma",
    "itiraf", "yüzleşme", "tartışma", "sorgu", "sorgulama",
    "keşif", "arama", "bulma", "gizem", "sır", "araştırma", "soruşturma",
]

NER_LABELS = ["O", "B-MEKAN", "I-MEKAN", "B-ZAMAN", "I-ZAMAN", "B-OLAY", "I-OLAY"]

GENRE_MAPPING = {
    "Dram": "Dram", "Drama": "Dram", "Aile Draması": "Dram",
    "Sosyal Dram": "Dram", "Psikolojik Dram": "Dram",
    "Komedi": "Komedi", "Romantik Komedi": "Romantik", "Komedi/Dram": "Dram",
    "Romantik": "Romantik", "Aşk": "Romantik",
    "Gerilim/Suç": "Polisiye", "Suç/Gerilim": "Polisiye", "Polisiye": "Polisiye",
    "Gerilim": "Polisiye", "Suç": "Polisiye", "Gizem": "Polisiye",
    "Aksiyon": "Aksiyon", "Macera": "Aksiyon", "Aksiyon/Macera": "Aksiyon",
    "Tarihi": "Tarihi", "Dönem Draması": "Tarihi", "Tarihi Dram": "Tarihi",
    "Tarihi/Dram": "Tarihi",
    "Fantastik": "Fantastik", "Bilim Kurgu": "Fantastik", "Korku": "Fantastik",
}


def turkish_lower(s):
    if pd.isna(s):
        return ""
    return str(s).replace("İ", "i").replace("I", "ı").lower().strip()


def normalize_title(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s*\(.*?\)", "", s)
    s = re.sub(r"\s*\[aug\d+\]$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def bio_tags(sentence):
    tokens = sentence.strip().split()
    tags = ["O"] * len(tokens)
    for i, token in enumerate(tokens):
        low = turkish_lower(token).rstrip(".,;:!?)(\"'")
        if low in PLACE_TERMS or any(kw in low for kw in PLACE_TERMS if len(kw) > 4):
            if i > 0 and tokens[i - 1][0].isupper() and tags[i - 1] == "O":
                tags[i - 1] = "B-MEKAN"
                tags[i] = "I-MEKAN"
            else:
                tags[i] = "B-MEKAN"
        elif low in TIME_TERMS:
            tags[i] = "B-ZAMAN"
        elif re.match(r"^\d{4}", token):
            tags[i] = "B-ZAMAN"
        elif low in EVENT_TERMS or any(kw in low for kw in EVENT_TERMS if len(kw) > 5):
            tags[i] = "B-OLAY"
    return tokens, tags


def map_genre(raw):
    if pd.isna(raw):
        return None
    g = str(raw).strip()
    if g in GENRE_MAPPING:
        return GENRE_MAPPING[g]
    if "/" in g:
        head = g.split("/")[0].strip()
        if head in GENRE_MAPPING:
            return GENRE_MAPPING[head]
    t = g.lower()
    if "polisiye" in t or "suç" in t or "gerilim" in t:
        return "Polisiye"
    if "aksiyon" in t or "macera" in t:
        return "Aksiyon"
    if "tarihi" in t or "dönem" in t:
        return "Tarihi"
    if "romantik" in t or "aşk" in t:
        return "Romantik"
    if "komedi" in t:
        return "Komedi"
    if "dram" in t:
        return "Dram"
    if "fantastik" in t or "bilim" in t or "korku" in t:
        return "Fantastik"
    return None


def augment_minority_classes(df, min_examples, seed=SEED):
    rows = []
    for label in sorted(df["label"].unique()):
        subset = df[df["label"] == label]
        if len(subset) == 0 or len(subset) >= min_examples:
            continue
        missing = min_examples - len(subset)
        subset = subset.sample(frac=1, random_state=seed)
        new_rows = []
        for _, row in subset.iterrows():
            if len(new_rows) >= missing:
                break
            sentences = [c.strip() for c in re.split(r"(?<=[.!?])\s+", str(row["senaryo_ozeti"]))
                         if len(c.strip()) > 30]
            if len(sentences) < 4:
                continue
            mid = len(sentences) // 2
            for tag, part in (("aug1", sentences[:mid]), ("aug2", sentences[mid:])):
                new_rows.append({
                    "yapim_adi": f"{row['yapim_adi']} [{tag}]",
                    "senaryo_ozeti": " ".join(part),
                    "label": label,
                    "yapim_yili": row.get("yapim_yili"),
                    "_parent": row["_parent"],
                    "is_augmented": True,
                })
        rows.extend(new_rows[:missing])
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True) if rows else df.copy()


def grouped_stratified_split(df, test_frac=0.15, val_frac=0.10, seed=SEED):
    rng = np.random.RandomState(seed)
    groups = df.groupby("_parent")["label"].first().reset_index()
    train_g, val_g, test_g = [], [], []
    for label in sorted(groups["label"].unique()):
        g = groups[groups["label"] == label]["_parent"].values.copy()
        rng.shuffle(g)
        n = len(g)
        n_test = max(1, int(round(n * test_frac))) if n >= 3 else (1 if n >= 2 else 0)
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
        n_test = min(n_test, max(0, n - 1))
        n_val = min(n_val, max(0, n - n_test - 1))
        test_g.extend(g[:n_test])
        val_g.extend(g[n_test:n_test + n_val])
        train_g.extend(g[n_test + n_val:])
    part = lambda keys: df[df["_parent"].isin(set(keys))].copy()
    return part(train_g), part(val_g), part(test_g)


def prepare_genre(summaries, productions, out_dir, min_examples=50):
    summaries = summaries.copy()
    productions = productions.copy()
    summaries["_norm"] = summaries["yapim_adi"].apply(normalize_title)
    productions["_norm"] = productions["yapim_adi"].apply(normalize_title)

    merged = summaries.merge(
        productions[["_norm", "tur", "yapim_yili"]].drop_duplicates("_norm"),
        on="_norm", how="left")
    merged["label"] = merged["tur"].apply(map_genre)
    df = merged[merged["label"].notna()][
        ["yapim_adi", "senaryo_ozeti", "label", "yapim_yili", "_norm"]].copy()
    df = df[df["senaryo_ozeti"].fillna("").apply(len) >= 100]
    df = df.drop_duplicates(subset=["senaryo_ozeti"]).reset_index(drop=True)
    df = df.rename(columns={"_norm": "_parent"})
    df["is_augmented"] = False

    train, val, test = grouped_stratified_split(df)
    n_before = len(train)
    train = augment_minority_classes(train, min_examples)

    assert not (set(train["_parent"]) & set(test["_parent"]))
    assert not (set(train["_parent"]) & set(val["_parent"]))
    assert not (set(val["_parent"]) & set(test["_parent"]))

    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    for part in (train, val, test):
        part["label_id"] = part["label"].map(label2id)
    train = train.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "genre_train.csv", index=False, encoding="utf-8")
    val.to_csv(out_dir / "genre_val.csv", index=False, encoding="utf-8")
    test.to_csv(out_dir / "genre_test.csv", index=False, encoding="utf-8")
    with open(out_dir / "genre_label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": {v: k for k, v in label2id.items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"genre: train={len(train)} ({len(train) - n_before} augmented) "
          f"val={len(val)} test={len(test)} classes={labels}")


def prepare_ner(pairs, summaries, out_dir, ner_summaries=None):
    if ner_summaries is not None:
        summaries = ner_summaries
    examples = []
    for _, row in pairs.iterrows():
        scene = str(row["sahne_bilgisi"]).strip()
        if len(scene) < 15:
            continue
        tokens, tags = bio_tags(scene)
        if tokens:
            examples.append({"tokens": tokens, "ner_tags": tags,
                             "kaynak": "sahne_bilgisi", "metin": scene})
    for _, row in summaries.iterrows():
        for sentence in re.split(r"(?<=[.!?])\s+|\r\n", str(row["senaryo_ozeti"]).strip()):
            sentence = sentence.strip()
            if len(sentence) < 40:
                continue
            tokens, tags = bio_tags(sentence)
            if tokens:
                examples.append({"tokens": tokens, "ner_tags": tags,
                                 "kaynak": "senaryo_ozeti", "metin": sentence})

    random.shuffle(examples)
    n = len(examples)
    n_train, n_val = int(n * 0.80), int(n * 0.10)
    splits = {"train": examples[:n_train],
              "dev": examples[n_train:n_train + n_val],
              "test": examples[n_train + n_val:]}

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out_dir / f"ner_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "ner_labels.json", "w", encoding="utf-8") as f:
        json.dump({"labels": NER_LABELS,
                   "label2id": {l: i for i, l in enumerate(NER_LABELS)},
                   "id2label": {i: l for i, l in enumerate(NER_LABELS)}},
                  f, ensure_ascii=False, indent=2)
    print(f"ner: train={len(splits['train'])} dev={len(splits['dev'])} test={len(splits['test'])}")


def prepare_retrieval(pairs, locations, out_dir):
    profiles = dict(zip(locations["id"], locations["profile"]))
    positives = []
    for _, row in pairs.iterrows():
        scene = str(row["sahne_bilgisi"]).strip()
        profile = profiles.get(row["lokasyon_id"], "")
        if not scene or not profile or len(scene) < 15:
            continue
        positives.append({"query": scene, "passage": profile,
                          "lokasyon_id": int(row["lokasyon_id"])})

    random.shuffle(positives)
    n = len(positives)
    n_test = max(100, int(n * 0.10))
    n_val = max(50, int(n * 0.10))
    splits = {"test": positives[:n_test],
              "val": positives[n_test:n_test + n_val],
              "train": positives[n_test + n_val:]}

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out_dir / f"sbert_{name}.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"retrieval: train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])} test locations={len(set(r['lokasyon_id'] for r in splits['test']))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=str(ROOT / "data/catalogue"))
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--ner-summaries", default=None,
                    help="CSV with a senaryo_ozeti column to use for the summary-derived NER "
                         "sentences (default: the catalogue summaries)")
    a = ap.parse_args()

    cat = Path(a.catalogue)
    out = Path(a.out)
    read = lambda name: pd.read_csv(cat / name, encoding="utf-8-sig")
    summaries = read("senaryo_ozetleri.csv")
    locations = read("lokasyonlar.csv")
    productions = read("yapimlar.csv")
    pairs = read("yapim_lokasyon.csv")

    locations["profile"] = locations.apply(
        lambda r: f"{r['lokasyon_adi']} {r['sehir']} {r['ilce']} {r['populer_aktiviteler']}", axis=1)

    ner_summaries = None
    if a.ner_summaries:
        ner_summaries = pd.read_csv(a.ner_summaries, encoding="utf-8-sig")

    prepare_genre(summaries, productions, out / "genre")
    prepare_ner(pairs, summaries, out / "ner", ner_summaries=ner_summaries)
    prepare_retrieval(pairs, locations, out / "retrieval")


if __name__ == "__main__":
    main()
