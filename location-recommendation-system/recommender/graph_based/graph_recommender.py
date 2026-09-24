import collections
import math

import pandas as pd


class SimilarLocationIndex:
    def __init__(self, pairs, productions=None, locations=None):
        self.weights = collections.defaultdict(float)
        self.degree = collections.Counter()
        self.location_productions = collections.defaultdict(set)
        self.genre_vectors = {}
        self.province = {}
        self._build(pairs, productions, locations)

    @classmethod
    def from_csv(cls, pairs_csv, productions_csv=None, locations_csv=None):
        read = lambda p: pd.read_csv(p, encoding="utf-8-sig") if p else None
        return cls(read(pairs_csv), read(productions_csv), read(locations_csv))

    def _build(self, pairs, productions, locations):
        by_production = collections.defaultdict(set)
        for _, r in pairs.iterrows():
            by_production[r["yapim_id"]].add(int(r["lokasyon_id"]))
            self.location_productions[int(r["lokasyon_id"])].add(r["yapim_id"])
        for loc, prods in self.location_productions.items():
            self.degree[loc] = len(prods)

        for locs in by_production.values():
            locs = sorted(locs)
            if len(locs) < 2:
                continue
            contribution = 1.0 / (len(locs) - 1)
            for i in range(len(locs)):
                for j in range(i + 1, len(locs)):
                    self.weights[(locs[i], locs[j])] += contribution
                    self.weights[(locs[j], locs[i])] += contribution

        if locations is not None and "sehir" in locations.columns:
            self.province = dict(zip(locations["id"].astype(int),
                                     locations["sehir"].fillna("").astype(str)))
        if productions is not None and "tur" in productions.columns:
            self._build_genre_vectors(productions)

    def _build_genre_vectors(self, productions):
        genre_of = dict(zip(productions["yapim_id"], productions["tur"].fillna("")))
        counts = collections.defaultdict(collections.Counter)
        for loc, prods in self.location_productions.items():
            for p in prods:
                g = str(genre_of.get(p, "")).strip()
                if g:
                    counts[loc][g] += 1
        df = collections.Counter()
        for c in counts.values():
            for g in c:
                df[g] += 1
        n = max(1, len(counts))
        for loc, c in counts.items():
            total = sum(c.values())
            v = {g: (cnt / total) * math.log(n / (1 + df[g])) for g, cnt in c.items()}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            self.genre_vectors[loc] = {g: x / norm for g, x in v.items()}

    def genre_similarity(self, l, m):
        a, b = self.genre_vectors.get(l), self.genre_vectors.get(m)
        if not a or not b:
            return 0.0
        return sum(a[g] * b.get(g, 0.0) for g in a)

    def similarity(self, l, m):
        w = self.weights.get((l, m), 0.0)
        if w == 0.0:
            return 0.0
        return w / math.sqrt(max(1, self.degree[l]) * max(1, self.degree[m]))

    def recommend(self, seeds, k=10, w_genre=0.3, w_province=0.1, exclude=None):
        if isinstance(seeds, int):
            seeds = [seeds]
        exclude = set(exclude or []) | set(seeds)
        scores = collections.defaultdict(float)
        for s in seeds:
            for (a, b) in self.weights:
                if a != s or b in exclude:
                    continue
                base = self.similarity(a, b)
                if base == 0.0:
                    continue
                score = base
                if w_genre:
                    score += w_genre * self.genre_similarity(a, b) * base
                if w_province and self.province:
                    if self.province.get(a) and self.province.get(a) == self.province.get(b):
                        score += w_province * base
                scores[b] += score
        return sorted(scores.items(), key=lambda x: -x[1])[:k]

    @property
    def n_edges(self):
        return len(self.weights) // 2
