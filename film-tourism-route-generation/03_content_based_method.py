"""
§3.2.2 CONTENT-BASED ROUTE GENERATION
========================================
Tür (TF-IDF) + tema (anahtar kelime) hibrit içerik skoru, coğrafi yakınlık skoru, ve
istatistiksel eşikle değişken durak sayısı seçimi. Üç örnek senaryo (Tablo 3) ile
doğrulanır.

§3.2.1'deki solve_route() fonksiyonunu (TSP sıralaması) değiştirmeden yeniden kullanır.
Bu modülün generate_hybrid_route() fonksiyonu, §3.2.4 tarafından yeniden kullanılır.
"""

import numpy as np
import pandas as pd

import importlib.util


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sp = _load_module('02_shortest_path_method.py', 'sp_module')
haversine_km = _sp.haversine_km
solve_route = _sp.solve_route


THEME_KEYWORDS = {
    'tarihi': ['tarih'], 'doğal': ['doğa', 'vadi', 'mağara', 'yürüyüş'],
    'kültürel': ['kültür', 'sanat', 'müze'],
    'gastronomi': ['gastronomi', 'kafe', 'restoran', 'lezzet'],
    'gece_hayati': ['gece hayatı', 'bar', 'eğlence'],
    'macera': ['macera', 'balon', 'tekne', 'at binme'],
}


def build_features(df):
    """Lokasyon x Tür TF-IDF matrisi ve Lokasyon x Tema ikili (binary) matrisi kurar."""
    GENRES = sorted(df['tur'].unique())
    loc = (df[['lokasyon_id', 'lokasyon_adi', 'sehir', 'ilce', 'lat', 'lon',
                'populer_aktiviteler']]
           .drop_duplicates(subset='lokasyon_id').reset_index(drop=True)
           .sort_values('lokasyon_id').reset_index(drop=True))
    N = len(loc)

    counts = (df.groupby('lokasyon_id')['tur'].value_counts().unstack(fill_value=0)
              .reindex(columns=GENRES, fill_value=0).reindex(loc['lokasyon_id']).fillna(0)
              .to_numpy(dtype=float))
    rs = counts.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    tf = counts / rs
    doc_freq = (counts > 0).sum(axis=0); doc_freq[doc_freq == 0] = 1
    idf = np.log(N / doc_freq) + 1
    tfidf = tf * idf
    norm = np.linalg.norm(tfidf, axis=1, keepdims=True); norm[norm == 0] = 1
    tfidf_normed = tfidf / norm

    act = loc['populer_aktiviteler'].fillna('').str.lower()
    theme_names = list(THEME_KEYWORDS.keys())
    theme_matrix = np.zeros((N, len(theme_names)))
    for j, t in enumerate(theme_names):
        theme_matrix[:, j] = act.str.contains('|'.join(THEME_KEYWORDS[t]), regex=True).astype(float)

    return loc, GENRES, tfidf_normed, theme_matrix, theme_names


def generate_hybrid_route(df, features, genre_weights=None, theme_interests=None,
                           ref_region=None, geo_decay_km=20.0, min_stops=3, max_stops=10):
    """Hibrit skor = içerik_skoru x coğrafya_skoru; değişken-N istatistiksel eşikle seçim."""
    loc, GENRES, tfidf_normed, theme_matrix, theme_names = features
    N = len(loc)

    content_score = np.zeros(N); n_comp = 0
    if genre_weights:
        v = np.array([genre_weights.get(g, 0.0) for g in GENRES])
        if np.linalg.norm(v) > 0:
            content_score += tfidf_normed @ (v / np.linalg.norm(v)); n_comp += 1
    if theme_interests:
        idxs = [theme_names.index(t) for t in theme_interests if t in theme_names]
        if idxs:
            content_score += theme_matrix[:, idxs].mean(axis=1); n_comp += 1
    if n_comp > 1: content_score /= n_comp
    if n_comp == 0: content_score[:] = 1.0

    ref = loc[loc['sehir'] == ref_region]
    ref_lat, ref_lon = ref['lat'].mean(), ref['lon'].mean()
    dists = np.array([haversine_km(ref_lat, ref_lon, r.lat, r.lon) for r in loc.itertuples()])
    geo_score = np.exp(-dists / geo_decay_km)
    hybrid = content_score * geo_score

    result = loc.copy(); result['hybrid_score'] = hybrid
    result = result.sort_values('hybrid_score', ascending=False)
    nz = result[result['hybrid_score'] > 0]['hybrid_score']
    threshold = (nz.mean() + nz.std()) if len(nz) else 0
    selected = result[result['hybrid_score'] >= threshold]
    if len(selected) < min_stops: selected = result.head(min_stops)
    elif len(selected) > max_stops: selected = result.head(max_stops)
    return selected, threshold


if __name__ == '__main__':
    df = pd.read_csv('merged_dataset.csv')
    df = df[df['yapim_adi'].notnull()].copy()
    features = build_features(df)

    scenarios = [
        ("Tarihi", dict(theme_interests=['tarihi']), 'Nevşehir', 20.0, 0.2051, 4, 34.02),
        ("Fantastik/Macera/Bilim Kurgu", dict(genre_weights={'Fantastik': 1.0, 'Macera': 0.7, 'Bilim Kurgu': 0.5}),
         'İstanbul', 15.0, 0.1910, 5, 19.44),
        ("Dram/Komedi", dict(genre_weights={'Dram': 1.0, 'Komedi': 0.6}), 'İzmir', 20.0, 0.0961, 10, 60.52),
    ]

    for label, kwargs, region, decay, exp_threshold, exp_n, exp_dist in scenarios:
        print(f"\n=== Senaryo: {label} ({region}) ===")
        selected, threshold = generate_hybrid_route(df, features, ref_region=region,
                                                      geo_decay_km=decay, **kwargs)
        route, total = solve_route(selected)
        print(f"Eşik: {threshold:.4f} (beklenen: {exp_threshold})")
        print(f"Durak: {len(route)} (beklenen: {exp_n})")
        print(f"Mesafe: {total:.2f} km (beklenen: {exp_dist})")
        for name in route['lokasyon_adi']:
            print(f"  - {name}")
        assert len(route) == exp_n, f"{label}: durak sayısı uyuşmuyor!"
        assert abs(total - exp_dist) < 0.05, f"{label}: mesafe uyuşmuyor!"

    print("\n✓ Tüm doğrulamalar geçti.")
