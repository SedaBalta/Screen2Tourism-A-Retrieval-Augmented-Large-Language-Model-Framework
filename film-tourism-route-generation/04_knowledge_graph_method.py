"""
§3.2.3 KNOWLEDGE GRAPH-BASED ROUTE GENERATION
================================================
Heterojen bilgi grafı (Film, Oyuncu, Yönetmen, Lokasyon, Tür düğümleri) + Node2Vec implementasyonu yapılmıştır.
Aynı fantastik/macera/bilim-kurgu profiliyle §3.2.2 ile doğrudan karşılaştırılabilir bir İstanbul rotası üretilmiştir. (Tablo 6).

§3.2.1'deki solve_route() fonksiyonu yeniden kullanılmaktadır.
"""

import re
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

import importlib.util


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sp = _load_module('02_shortest_path_method.py', 'sp_module')
haversine_km = _sp.haversine_km
solve_route = _sp.solve_route

random.seed(42)
np.random.seed(42)


def clean_actor_list(raw):
    if pd.isnull(raw) or str(raw).strip().lower() == 'bilinmiyor':
        return []
    text = re.sub(r'\{\{.*?liste,?\s*,?', '', str(raw))
    parts = [p.strip(' *') for p in text.split(',')]
    return [p for p in parts if p and p.lower() != 'bilinmiyor']


def build_knowledge_graph(df, near_threshold_km=15):
    G = nx.Graph()
    films = df.drop_duplicates(subset='yapim_id')[['yapim_id', 'yapim_adi', 'tur', 'yonetmen', 'oyuncular']]
    for row in films.itertuples():
        fn = f'FILM_{row.yapim_id}'
        G.add_node(fn, node_type='Film', label=row.yapim_adi)
        if pd.notnull(row.tur):
            gn = f'GENRE_{row.tur}'; G.add_node(gn, node_type='Genre', label=row.tur); G.add_edge(fn, gn)
        if pd.notnull(row.yonetmen) and row.yonetmen.strip().lower() != 'bilinmiyor':
            dn = f'DIR_{row.yonetmen.strip()}'
            G.add_node(dn, node_type='Director', label=row.yonetmen.strip()); G.add_edge(fn, dn)
        for actor in clean_actor_list(row.oyuncular):
            an = f'ACTOR_{actor}'; G.add_node(an, node_type='Actor', label=actor); G.add_edge(fn, an)

    locations = df.drop_duplicates(subset='lokasyon_id')[['lokasyon_id', 'lokasyon_adi', 'sehir', 'ilce', 'lat', 'lon']]
    for row in locations.itertuples():
        G.add_node(f'LOC_{row.lokasyon_id}', node_type='Location', label=row.lokasyon_adi,
                   sehir=row.sehir, ilce=row.ilce, lat=row.lat, lon=row.lon)
    for row in df.itertuples():
        G.add_edge(f'FILM_{row.yapim_id}', f'LOC_{row.lokasyon_id}', relation='ÇEKİLDİ')

    loc_list = list(locations.itertuples())
    for i in range(len(loc_list)):
        for j in range(i + 1, len(loc_list)):
            d = haversine_km(loc_list[i].lat, loc_list[i].lon, loc_list[j].lat, loc_list[j].lon)
            if d <= near_threshold_km:
                G.add_edge(f'LOC_{loc_list[i].lokasyon_id}', f'LOC_{loc_list[j].lokasyon_id}', relation='YAKIN')
    return G


def node2vec_embeddings(G, num_walks=8, walk_length=30, window=5, embed_dim=64, p=1.0, q=1.0):
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    adj = {n: list(G.neighbors(n)) for n in nodes}

    def walk(start):
        w = [start]
        while len(w) < walk_length:
            cur = w[-1]; nbrs = adj[cur]
            if not nbrs: break
            if len(w) == 1:
                w.append(random.choice(nbrs))
            else:
                prev = w[-2]; pset = set(adj[prev])
                weights = [1.0/p if x == prev else (1.0 if x in pset else 1.0/q) for x in nbrs]
                tot = sum(weights); probs = [x/tot for x in weights]
                w.append(np.random.choice(nbrs, p=probs))
        return w

    walks = []
    for _ in range(num_walks):
        shuffled = nodes[:]; random.shuffle(shuffled)
        for node in shuffled:
            walks.append(walk(node))

    cooc = defaultdict(float)
    for w in walks:
        L = len(w)
        for i, wi_ in enumerate(w):
            wi = node_to_idx[wi_]
            for j in range(max(0, i-window), min(L, i+window+1)):
                if i == j: continue
                cooc[(wi, node_to_idx[w[j]])] += 1.0

    rows, cols, vals = zip(*[(k[0], k[1], v) for k, v in cooc.items()])
    C = sparse.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    total = C.sum()
    rsum = np.asarray(C.sum(axis=1)).flatten(); csum = np.asarray(C.sum(axis=0)).flatten()
    Cc = C.tocoo()
    ppmi_vals = [max(np.log((v*total)/(rsum[i]*csum[j]+1e-12)+1e-12), 0.0)
                 for i, j, v in zip(Cc.row, Cc.col, Cc.data)]
    PPMI = sparse.csr_matrix((ppmi_vals, (Cc.row, Cc.col)), shape=(n_nodes, n_nodes))
    embeddings = TruncatedSVD(n_components=embed_dim, random_state=42).fit_transform(PPMI)
    return nodes, node_to_idx, embeddings


def recommend_locations(G, nodes, node_to_idx, embeddings, genre_weights, region=None):
    def cosine(a, b):
        return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    vecs, weights = [], []
    for g, w in genre_weights.items():
        key = f'GENRE_{g}'
        if key in node_to_idx:
            vecs.append(embeddings[node_to_idx[key]]); weights.append(w)
    tourist_emb = np.average(vecs, axis=0, weights=weights)

    loc_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'Location']
    rows = []
    for n in loc_nodes:
        d = G.nodes[n]
        if region and d['sehir'] != region:
            continue
        sim = cosine(tourist_emb, embeddings[node_to_idx[n]])
        rows.append({'lokasyon_id': int(n.split('_')[1]), 'lokasyon_adi': d['label'],
                     'ilce': d['ilce'], 'lat': d['lat'], 'lon': d['lon'], 'sim': sim})
    return pd.DataFrame(rows).sort_values('sim', ascending=False)


if __name__ == '__main__':
    df = pd.read_csv('merged_dataset.csv')
    df = df[df['yapim_adi'].notnull()].copy()

    print("=== Bilgi grafı kuruluyor ===")
    G = build_knowledge_graph(df)
    counts = pd.Series([d['node_type'] for _, d in G.nodes(data=True)]).value_counts()
    print(counts)
    print(f"Toplam düğüm: {G.number_of_nodes()} (beklenen: 3748), "
          f"kenar: {G.number_of_edges()} (beklenen: 11670)")
    assert G.number_of_nodes() == 3748 and G.number_of_edges() == 11670, \
        "Graf boyutu beklenenle uyuşmuyor!"

    print("\n=== Node2Vec gömmeleri hesaplanıyor ===")
    nodes, node_to_idx, embeddings = node2vec_embeddings(G)

    print("\n=== Öneri: Fantastik/Macera/Bilim Kurgu profili, İstanbul ===")
    profile = {'Fantastik': 1.0, 'Macera': 0.7, 'Bilim Kurgu': 0.5}
    sims = recommend_locations(G, nodes, node_to_idx, embeddings, profile, region='İstanbul')
    top5 = sims.head(5)
    print(top5[['lokasyon_adi', 'sim']].to_string(index=False))

    route, total = solve_route(top5)
    print(f"\nTSP-sıralı rota mesafesi: {total:.2f} km (beklenen: ~30.1 km)")
    for name in route['lokasyon_adi']:
        print(f"  - {name}")

    expected_top5 = {'Kilyos', 'Sultanahmet', 'Beyoğlu', 'Zekeriyaköy Villaları', 'İstiklal Caddesi'}
    assert set(top5['lokasyon_adi']) == expected_top5, "Top-5 lokasyon kümesi beklenenle uyuşmuyor!"
    print("\n✓ Tüm doğrulamalar geçti (not: rastgele yürüyüş tabanlı olduğu için mesafe "
          "±birkaç km degiskenlik gösterebilir, lokasyon kümesi ve mertebe sabittir).")
