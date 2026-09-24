"""
§3.3 RAG TABANLI TURİZM ASİSTANI (RAG-Based Tourism Assistant)
==================================================================
Kapsam: yalnızca bilgi sorguları (Q&A) - "X nerede çekildi", "Y lokasyonunun sahne bilgisi ne" gibi serbest metin sorulara, veri setine dayandırılmış (grounded) cevap üretimi saglanmistir.

Mimari: Klasik BM25 (sparse, kütüphanesiz) retrieval + LLM tabanlı
§3.2.4'te öğrenilen ders burada yapısal bir kurala dönüştürülür.
"yalnızca getirilen belgelerden cevap ver, yoksa bilmediğini söyle" kısıtını icermektedir.
"""

import re
import math
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================================
# ADIM 1: DOKÜMAN DEPOSU (Document Store)
# ============================================================================
# Her doküman bir (yapım, lokasyon) çiftini temsil eder - veri setindeki tüm ilgili
# alanları (yapım adı/türü/yönetmeni, lokasyon adı/ilçe/şehri, sahne bilgisi, popüler
# aktiviteler) tek bir metin parçasında birleştirir. Bu, her cevabın doğrudan tek bir
# doğrulanabilir veri satırına (lokasyon_id + yapim_id) izlenebilmesini sağlar.

def build_corpus(df):
    df = df[df['yapim_adi'].notnull()].copy()
    docs = []
    for row in df.itertuples():
        text = (
            f"Yapım: {row.yapim_adi} ({row.tur}, yönetmen: {row.yonetmen}). "
            f"Lokasyon: {row.lokasyon_adi}, {row.ilce}, {row.sehir}. "
            f"Sahne bilgisi: {row.sahne_bilgisi}. "
            f"Popüler aktiviteler: {row.populer_aktiviteler}. "
            f"Turizm potansiyeli: {row.turizm_potansiyeli}."
        )
        docs.append({
            'doc_id': f"{row.yapim_id}_{row.lokasyon_id}",
            'yapim_id': row.yapim_id, 'yapim_adi': row.yapim_adi,
            'lokasyon_id': row.lokasyon_id, 'lokasyon_adi': row.lokasyon_adi,
            'text': text,
        })
    return pd.DataFrame(docs)


# ============================================================================
# ADIM 2: BM25 RETRIEVER 
# ============================================================================

TURKISH_STOPWORDS = {'ve', 'ile', 'bir', 'bu', 'da', 'de', 'için', 'gibi', 'çok', 'daha',
                      'olan', 'olarak', 'the', 'a', 'an', 'in', 'of', 'to'}


def tokenize(text):
    text = re.sub(r'[^\wçğıöşüÇĞİÖŞÜ\s]', ' ', str(text).lower())
    return [t for t in text.split() if t and t not in TURKISH_STOPWORDS]


class BM25:
    """Robertson & Sparse Jones BM25 - klasik sparse retrieval, k1=1.5, b=0.75."""

    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tokenized = [tokenize(d) for d in documents]
        self.doc_len = np.array([len(d) for d in self.tokenized])
        self.avgdl = self.doc_len.mean()
        self.N = len(documents)

        self.df = Counter()
        for doc in self.tokenized:
            for term in set(doc):
                self.df[term] += 1
        self.idf = {t: math.log((self.N - df + 0.5) / (df + 0.5) + 1) for t, df in self.df.items()}

        self.tf = [Counter(doc) for doc in self.tokenized]

    def score(self, query):
        q_terms = tokenize(query)
        scores = np.zeros(self.N)
        for term in q_terms:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i in range(self.N):
                f = self.tf[i].get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores

    def retrieve(self, query, top_k=5):
        scores = self.score(query)
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


# ============================================================================
# ADIM 3: KISITLANMIŞ ÜRETİM (Generation) - LLM entegrasyon noktası
# ============================================================================
# NOT: Aşağıdaki sistem promptu, gerçek bir LLM API çağrısında (örn. Claude, GPT) system
# prompt olarak kullanılmak üzere tasarlanmıştır. 

RAG_SYSTEM_PROMPT = """Sen bir film turizmi asistanısın. Sana verilen BELGELER dışında
hiçbir bilgi kullanma. Eğer belgelerde soruyu cevaplayacak bilgi yoksa, KESİNLİKLE
uydurma - "Bu bilgi veri setimde yok" de. Her cevabında hangi belgeden (yapım/lokasyon)
yararlandığını belirt."""


def build_generation_prompt(query, retrieved_docs, corpus):
    context = "\n\n".join(
        f"[Belge {i+1} - {corpus.iloc[idx]['yapim_adi']} / {corpus.iloc[idx]['lokasyon_adi']}]\n"
        f"{corpus.iloc[idx]['text']}"
        for i, (idx, score) in enumerate(retrieved_docs)
    )
    return f"{RAG_SYSTEM_PROMPT}\n\nBELGELER:\n{context}\n\nSORU: {query}\n\nCEVAP:"


if __name__ == '__main__':
    df = pd.read_csv('merged_dataset.csv')
    corpus = build_corpus(df)
    print(f"Doküman deposu kuruldu: {len(corpus)} belge "
          f"(beklenen: 1752, yapim_adi dolu satır sayısı kadar)")
    assert len(corpus) == 1752

    bm25 = BM25(corpus['text'].tolist())
    print(f"BM25 indeksi kuruldu: {bm25.N} doküman, {len(bm25.idf)} benzersiz terim")

    # Hızlı duman testi
    query = "Sadakatsiz dizisi hangi lokasyonlarda çekildi?"
    results = bm25.retrieve(query, top_k=5)
    print(f"\nÖrnek sorgu: \"{query}\"")
    for idx, score in results:
        row = corpus.iloc[idx]
        print(f"  [{score:.2f}] {row['yapim_adi']} @ {row['lokasyon_adi']}")

    print("\n✓ Doküman deposu ve BM25 indeksi hazır.")
