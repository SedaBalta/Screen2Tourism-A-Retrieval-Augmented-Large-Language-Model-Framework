import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SYSTEM = """Sen bir sinema prodüksiyonunda çalışan kıdemli mekan sorumlususun \
(location scout). Bir yapımcı sana sahne tanımını veriyor ve sen bu sahnenin \
çekilebileceği ALTERNATİF mekanlar arıyorsun.

GÖREVİN: Önerilen mekanın, sahnenin ihtiyaç duyduğu mekan TÜRÜNÜ ve \
ATMOSFERİNİ sağlayıp sağlamadığını değerlendirmek.

EN ÖNEMLİ KURAL — İKAME MANTIGI:
Sahne tanımında geçen yer adı, o sahnenin çekilmesi ZORUNLU olan tek yer \
değildir; yalnızca aranan mekan karakterini tarif eder. Farklı bir şehirdeki \
veya bölgedeki bir mekan, aynı türde ve atmosferde olduğu sürece GEÇERLİ bir \
alternatiftir. "Sahnede X yazıyor ama bu mekan Y'de" gerekçesiyle puan DÜŞÜRME.

Değerlendirme ölçeği:
2 = Uygun. Mekan türü ve atmosfer sahneyle örtüşüyor. Sahne burada çekilebilir.
    (Yer adı farklı olsa bile, karakter aynıysa 2 ver.)
1 = Kısmen uygun. Mekan türü doğru fakat belirgin bir çekince var: ölçek çok \
    farklı, dönem/mimari uyumsuz, ya da atmosfer kısmen karşılanıyor.
0 = Uygun değil. Mekan TÜRÜ sahneyle çelişiyor. \
    (Örn: sahne sahil istiyor, mekan fabrika. Sahne dar sokak istiyor, \
    mekan açık ova.)

Örnekler:
- Sahne: "Bodrum Kalesi'ndeki kale sahneleri" / Mekan: "Alanya Kalesi, Antalya"
  -> 2. İkisi de deniz kenarında tarihi kale; şehir farkı önemli değil.
- Sahne: "Kars'taki eski Rus evleri" / Mekan: "Kozan Tarihi Evleri, Adana"
  -> 1. İkisi de tarihi ev dokusu, fakat mimari gelenek belirgin biçimde farklı.
- Sahne: "Karadeniz yaylasında sisli sabah" / Mekan: "Marmaris Yat Limanı"
  -> 0. Mekan türü tamamen farklı.

Yanıtını yalnızca şu JSON biçiminde ver:
{"skor": 0|1|2, "gerekce": "tek cümlelik kısa gerekçe"}"""


def judge(client, model, scene, row, retries=4):
    user = (f"SAHNE TANIMI:\n{scene}\n\nÖNERİLEN MEKAN:\n"
            f"Ad: {row['location_name']}\n"
            f"İl / İlçe: {row['province']} / {row.get('district', '')}\n"
            f"Popüler aktiviteler: {row.get('activities', '')}")
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                response_format={"type": "json_object"})
            data = json.loads((response.choices[0].message.content or "").strip())
            score = int(data.get("skor", 0))
            return (score if score in (0, 1, 2) else 0), str(data.get("gerekce", ""))[:200]
        except Exception as e:
            if attempt == retries - 1:
                print(f"failed: {type(e).__name__}")
                return None, ""
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "data/annotation/annotation_set.xlsx"))
    ap.add_argument("--model", required=True, help="judge model name")
    ap.add_argument("--human-subset", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "data/annotation/annotation_llm.xlsx"))
    ap.add_argument("--checkpoint", default=str(ROOT / "data/annotation/llm_annotate_checkpoint.json"))
    a = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    df = pd.read_excel(a.file)
    ckpt_path = Path(a.checkpoint)
    ckpt = json.load(open(ckpt_path, encoding="utf-8")) if ckpt_path.exists() else {}
    print(f"{len(df)} pairs; {len(ckpt)} already judged")

    scores, reasons = [], []
    for i, row in df.iterrows():
        key = str(i)
        if key not in ckpt:
            score, why = judge(client, a.model, row["scene_description"], row)
            ckpt[key] = [0 if score is None else score, why or ("judge failed" if score is None else "")]
            if (i + 1) % 20 == 0:
                json.dump(ckpt, open(ckpt_path, "w", encoding="utf-8"), ensure_ascii=False)
                print(f"  {i + 1}/{len(df)}")
        scores.append(ckpt[key][0])
        reasons.append(ckpt[key][1])
    json.dump(ckpt, open(ckpt_path, "w", encoding="utf-8"), ensure_ascii=False)

    df["llm_score"] = scores
    df["llm_reason"] = reasons
    df["annotator_1"] = scores
    df["annotator_2"] = scores
    df["judge_model"] = a.model
    df.to_excel(a.out, index=False)
    print(f"wrote {a.out}")
    print(df["llm_score"].value_counts().sort_index().to_string())

    if a.human_subset:
        random.seed(42)
        idx = sorted(random.sample(range(len(df)), min(a.human_subset, len(df))))
        subset = df.loc[idx, ["query_id", "scene_description", "candidate_id", "location_name",
                              "province", "district", "activities"]].copy()
        subset["human_score"] = ""
        subset["_row"] = idx
        subset_path = Path(a.out).with_name("human_subset.xlsx")
        subset.to_excel(subset_path, index=False)
        print(f"wrote {subset_path} ({len(subset)} rows) for expert assessment")


if __name__ == "__main__":
    main()
