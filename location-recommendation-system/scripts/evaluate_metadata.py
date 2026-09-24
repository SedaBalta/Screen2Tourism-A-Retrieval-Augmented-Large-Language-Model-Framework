import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.text_utils import turkish_lower  # noqa: E402

SUFFIXES = ["larında", "lerinde", "ndaki", "daki", "deki", "ları", "leri",
            "dan", "den", "tan", "ten", "nın", "nin", "da", "de", "ta", "te",
            "ya", "ye", "yi", "yı", "i", "ı", "u", "ü", "e", "a"]

CATEGORIES = {
    "park_doga": ["park", "bahçe", "orman", "yayla", "vadi", "dağ", "tepe"],
    "gar_ulasim": ["gar", "istasyon", "köprü", "liman", "iskele", "rıhtım"],
    "sokak_kentsel": ["sokak", "cadde", "mahalle", "meydan", "çarşı"],
    "konut_tarihi": ["konak", "köşk", "saray", "yalı", "kale", "han"],
    "is_endustri": ["fabrika", "depo", "sanayi", "atölye", "hangar"],
    "su_kenari": ["deniz", "sahil", "kıyı", "kumsal", "plaj", "göl", "nehir"],
    "sosyal_eglence": ["kafe", "restoran", "otel", "müze", "tiyatro"],
    "resmi_kurum": ["karakol", "adliye", "okul", "üniversite", "kışla"],
    "koy_kirsal": ["köy", "tarla", "çiftlik", "ahır", "mera"],
    "tarihi_antik": ["antik", "harabe", "kalıntı", "sit alanı", "surlar"],
    "ibadethane_inanc": ["cami", "kilise", "manastır", "türbe", "mescit"],
}

REGIONS = {
    "ege": ["izmir", "muğla", "aydın", "denizli", "manisa"],
    "karadeniz": ["trabzon", "rize", "artvin", "giresun", "ordu", "samsun"],
    "akdeniz": ["antalya", "mersin", "adana", "hatay", "isparta"],
    "marmara": ["istanbul", "bursa", "kocaeli", "çanakkale", "edirne"],
}


def word_match(text, keyword):
    t, k = turkish_lower(text), turkish_lower(keyword)
    if " " in k:
        return k in t
    return any(w == k or (w.startswith(k) and w[len(k):] in SUFFIXES) for w in re.findall(r"\w+", t))


def category_of(text):
    for c, kws in CATEGORIES.items():
        if any(word_match(text, k) for k in kws):
            return c
    return ""


def evaluate(test, locations, encoder_dir, category_boost=0.25):
    from sentence_transformers import SentenceTransformer

    id2city = dict(zip(locations["id"].astype(int), locations["sehir"].fillna("").astype(str)))
    passages = sorted(set(r["passage"] for r in test))
    index = {p: i for i, p in enumerate(passages)}
    gold = [index[r["passage"]] for r in test]
    queries = [r["query"] for r in test]

    cities = sorted({turkish_lower(c) for c in id2city.values() if c})
    passage_city = [next((c for c in cities if word_match(p, c)), "") for p in passages]
    passage_category = [category_of(p) for p in passages]

    model = SentenceTransformer(str(encoder_dir))
    enc = lambda texts: model.encode(texts, batch_size=32, convert_to_numpy=True,
                                     normalize_embeddings=True, show_progress_bar=False)
    P, Q = enc(passages), enc(queries)

    def run(geo, cat, k=5):
        hits = []
        for i, q in enumerate(queries):
            s = (Q[i] @ P.T).copy()
            if geo:
                target = next((c for c in cities if word_match(q, c)), "")
                if target:
                    s[[j for j in range(len(passages)) if passage_city[j] != target]] -= 1.0
                else:
                    for region, provinces in REGIONS.items():
                        if word_match(q, region):
                            s[[j for j in range(len(passages)) if passage_city[j] not in provinces]] -= 1.0
                            break
            if cat:
                active = {c for c, kws in CATEGORIES.items() if any(word_match(q, k_) for k_ in kws)}
                if active:
                    s[[j for j in range(len(passages)) if passage_category[j] in active]] += category_boost
            hits.append(1.0 if gold[i] in np.argsort(s)[::-1][:k] else 0.0)
        return np.array(hits)

    return {"No metadata filter": run(False, False),
            "Geographic only": run(True, False),
            "Category boost only": run(False, True),
            "Both": run(True, True)}


def paired_bootstrap(a, b, n_rep=10000, seed=42):
    rng = np.random.RandomState(seed)
    d = a - b
    observed = d.mean()
    centred = d - observed
    n = len(d)
    count = sum(abs(centred[rng.randint(0, n, n)].mean()) >= abs(observed) for _ in range(n_rep))
    return observed * 100, (count + 1) / (n_rep + 1)


def mcnemar(a, b):
    from scipy.stats import binomtest
    b01 = int(((a == 0) & (b == 1)).sum())
    b10 = int(((a == 1) & (b == 0)).sum())
    return 1.0 if b01 + b10 == 0 else binomtest(b10, b01 + b10, 0.5).pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/retrieval/sbert_test.json"))
    ap.add_argument("--locations", default=str(ROOT / "data/catalogue/lokasyonlar.csv"))
    ap.add_argument("--encoder", default=str(ROOT / "models/retriever_e5_small"))
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    test = json.load(open(a.data, encoding="utf-8"))
    locations = pd.read_csv(a.locations, encoding="utf-8-sig")
    configs = evaluate(test, locations, a.encoder)

    print(f"{'setting':<24}{'HR@5':>9}")
    for name, hits in configs.items():
        print(f"{name:<24}{hits.mean() * 100:>8.2f}%")

    print(f"\n{'comparison':<46}{'diff':>8}{'boot p':>9}{'McNemar p':>11}")
    tests = {}
    for x, y in combinations(configs, 2):
        diff, p_boot = paired_bootstrap(configs[x], configs[y], a.bootstrap)
        p_mcn = mcnemar(configs[x], configs[y])
        tests[f"{x} vs {y}"] = {"diff": diff, "p_bootstrap": p_boot, "p_mcnemar": p_mcn}
        print(f"{x + ' vs ' + y:<46}{diff:>+7.2f}%{p_boot:>9.4f}{p_mcn:>11.4f}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"hit_rate_at_5": {k: float(v.mean() * 100) for k, v in configs.items()},
                   "paired_tests": tests, "n_queries": len(test)},
                  open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
