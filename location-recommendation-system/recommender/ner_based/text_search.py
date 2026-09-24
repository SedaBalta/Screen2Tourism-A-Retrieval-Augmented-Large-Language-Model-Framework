import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .categories import (CATEGORY_KEYWORDS, GENRE_ADJUSTMENTS, active_categories,
                         category_label, location_category, region_provinces)
from .entity_extraction import EntityExtractor
from .text_utils import turkish_lower, turkish_word_match, words


def location_profile(row):
    parts = [row.get("lokasyon_adi", ""), row.get("sehir", ""),
             row.get("ilce", ""), row.get("populer_aktiviteler", "")]
    return " ".join(str(p) for p in parts if pd.notna(p) and str(p).strip()).strip()


class TextLocationSearch:
    def __init__(self, encoder_dir, locations_csv, ner_dir=None, alpha=0.6,
                 category_boost=0.25, genre_boost=0.15, genre_penalty=0.10,
                 adjustment_pool=100):
        self.alpha = alpha
        self.category_boost = category_boost
        self.genre_boost = genre_boost
        self.genre_penalty = genre_penalty
        self.adjustment_pool = adjustment_pool

        self.locations = pd.read_csv(locations_csv, encoding="utf-8-sig")
        self.locations["profile"] = self.locations.apply(location_profile, axis=1)
        self.provinces = sorted(self.locations["sehir"].dropna().astype(str).unique())

        self.encoder = SentenceTransformer(str(encoder_dir))
        self.embeddings = self.encoder.encode(
            self.locations["profile"].tolist(), convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False)
        self.vectorizer = TfidfVectorizer(lowercase=True)
        self.tfidf = self.vectorizer.fit_transform(
            [turkish_lower(p) for p in self.locations["profile"]])
        self.extractor = EntityExtractor(ner_dir) if ner_dir else None

    def detect_provinces(self, text):
        text_lower = turkish_lower(text)
        text_words = set(words(text))
        found = []
        for province in self.provinces:
            p = turkish_lower(province)
            if (" " in p and p in text_lower) or (" " not in p and p in text_words):
                found.append(province)
        return found or region_provinces(text)

    def query_text(self, text):
        if self.extractor is None:
            return text
        entities = self.extractor.extract(text)
        return entities if entities.strip() else text

    def search(self, text, k=5, genre=None, provinces=None):
        provinces = provinces if provinces is not None else self.detect_provinces(text)
        mask = (self.locations["sehir"].isin(provinces).values if provinces
                else np.ones(len(self.locations), dtype=bool))
        if not mask.any():
            mask = np.ones(len(self.locations), dtype=bool)
        idx = np.flatnonzero(mask)
        candidates = self.locations.iloc[idx].reset_index(drop=True)

        query = self.query_text(text)
        query_norm = turkish_lower(query)
        q_emb = self.encoder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        dense = self.embeddings[idx] @ q_emb
        sparse = cosine_similarity(self.vectorizer.transform([query_norm]), self.tfidf[idx])[0]
        scores = self.alpha * dense + (1 - self.alpha) * sparse

        for key in active_categories(query_norm):
            keywords = CATEGORY_KEYWORDS[key]
            for i, row in candidates.iterrows():
                loc_text = f"{row['lokasyon_adi']} {row.get('populer_aktiviteler', '')} {row['profile']}"
                if any(turkish_word_match(loc_text, kw) for kw in keywords):
                    scores[i] += self.category_boost

        pool = np.argsort(scores)[::-1][:min(self.adjustment_pool, len(candidates))]
        adjustment = GENRE_ADJUSTMENTS.get(genre) if genre else None
        if adjustment:
            for i in pool:
                row = candidates.iloc[i]
                cat = location_category(f"{row['lokasyon_adi']} {row['profile']}")
                if cat in adjustment["boost"]:
                    scores[i] += self.genre_boost
                elif cat in adjustment["penalty"]:
                    scores[i] -= self.genre_penalty
            pool = np.argsort(scores)[::-1][:len(pool)]

        results = []
        for i in pool[:k]:
            row = candidates.iloc[i]
            results.append({
                "id": int(row["id"]),
                "name": str(row["lokasyon_adi"]),
                "province": str(row["sehir"]),
                "district": str(row.get("ilce", "")),
                "category": category_label(f"{row['lokasyon_adi']} {row['profile']}"),
                "score": float(scores[i]),
            })
        return results
