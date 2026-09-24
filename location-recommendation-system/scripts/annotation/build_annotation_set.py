import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from recommender.text_search import location_profile  # noqa: E402

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/retrieval/sbert_test.json"))
    ap.add_argument("--locations", default=str(ROOT / "data/catalogue/lokasyonlar.csv"))
    ap.add_argument("--encoder", default=str(ROOT / "models/retriever_e5_small"))
    ap.add_argument("--n-queries", type=int, default=50)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "data/annotation/annotation_set"))
    a = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    from sentence_transformers import SentenceTransformer

    test = json.load(open(a.data, encoding="utf-8"))
    loc = pd.read_csv(a.locations, encoding="utf-8-sig")
    loc["profile"] = loc.apply(location_profile, axis=1)
    ids = loc["id"].astype(int).tolist()
    id2row = {v: i for i, v in enumerate(ids)}
    province = dict(zip(loc["id"].astype(int), loc["sehir"].fillna("").astype(str)))

    pool = [r for r in test if int(r["lokasyon_id"]) in id2row]
    by_province = {}
    for r in pool:
        by_province.setdefault(province[int(r["lokasyon_id"])], []).append(r)
    provinces = sorted(by_province)
    chosen, i = [], 0
    while len(chosen) < min(a.n_queries, len(pool)):
        p = provinces[i % len(provinces)]
        if by_province[p]:
            chosen.append(by_province[p].pop(random.randrange(len(by_province[p]))))
        i += 1
        if all(not v for v in by_province.values()):
            break
    print(f"sampled {len(chosen)} queries across {len(provinces)} provinces")

    model = SentenceTransformer(a.encoder)
    enc = lambda texts: model.encode(texts, batch_size=32, convert_to_numpy=True,
                                     normalize_embeddings=True, show_progress_bar=False)
    P = enc(loc["profile"].tolist())
    Q = enc([r["query"] for r in chosen])

    rows = []
    for qi, r in enumerate(chosen):
        order = np.argsort(Q[qi] @ P.T)[::-1].tolist()
        top = order[:a.topk]
        gold_row = id2row[int(r["lokasyon_id"])]
        if gold_row not in top:
            top.append(gold_row)
        random.shuffle(top)
        for j in top:
            rows.append({
                "query_id": qi,
                "scene_description": r["query"],
                "candidate_id": ids[j],
                "location_name": loc.iloc[j]["lokasyon_adi"],
                "province": loc.iloc[j]["sehir"],
                "district": loc.iloc[j].get("ilce", ""),
                "activities": loc.iloc[j].get("populer_aktiviteler", ""),
                "annotator_1": "",
                "annotator_2": "",
                "_is_gold": int(j == gold_row),
                "_system_rank": order.index(j) + 1,
            })

    df = pd.DataFrame(rows)
    df.to_excel(f"{a.out}.xlsx", index=False)
    json.dump({"seed": SEED, "n_queries": len(chosen), "topk": a.topk, "n_pairs": len(df),
               "query_location_ids": [r["lokasyon_id"] for r in chosen]},
              open(Path(a.out).with_name("annotation_meta.json"), "w", encoding="utf-8"), indent=2)
    print(f"wrote {a.out}.xlsx: {len(df)} query-candidate pairs")
    print("scores: 2 = suitable, 1 = partially suitable, 0 = unsuitable")


if __name__ == "__main__":
    main()
