"""
§3.2.1 SHORTEST-PATH BASED ROUTE GENERATION
=============================================
Dijkstra, A* ve Floyd-Warshall algoritmalarının sıfırdan implementasyonu; bölgeye göre parametrik k-NN grafı kurulumu; İstanbul için karşılaştırmalı benchmark; Sadakatsiz
(Balat->Moda) örnek olayı.

Bu modül bölüm §3.2.2, §3.2.3 ve §3.2.4 tarafından import edilerek yeniden kullanılmaltadır.
"""

import math
import heapq
import random
from itertools import permutations

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def haversine_matrix(coords):
    lat = np.radians(coords[:, 0]); lon = np.radians(coords[:, 1])
    dlat = lat[:, None] - lat[None, :]; dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371.0*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_region_graph(df, region, k=6):
    """Belirli bir şehre (region) sınırlandırılmış k-NN grafı kurar (§3.2.1.2 - ülke
    geneli grafta karşılaşılan outlier/köprü sorununu önlemek için bölge-sınırlama)."""
    locations = (
        df[df['sehir'] == region][['lokasyon_id', 'lokasyon_adi', 'ilce', 'lat', 'lon']]
        .drop_duplicates(subset='lokasyon_id').reset_index(drop=True)
        .sort_values('lokasyon_id').reset_index(drop=True)
    )
    n = len(locations)
    coords = locations[['lat', 'lon']].to_numpy()
    names = locations['lokasyon_adi'].to_numpy()
    dm = haversine_matrix(coords)

    def build(kk):
        adj = {i: {} for i in range(n)}
        for i in range(n):
            nearest = np.argsort(dm[i])[1:kk + 1]
            for j in nearest:
                j = int(j); adj[i][j] = dm[i, j]; adj[j][i] = dm[i, j]
        return adj

    def is_connected(adj):
        visited, stack = set(), [0]
        while stack:
            u = stack.pop()
            if u in visited: continue
            visited.add(u); stack.extend(adj[u].keys())
        return len(visited) == n

    adj = build(k)
    while not is_connected(adj):
        k += 2
        adj = build(k)
    return adj, coords, names, locations


def dijkstra(adj, source, target):
    dist = {source: 0.0}; prev = {}; visited = set()
    pq = [(0.0, source)]; nodes_expanded = 0
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u); nodes_expanded += 1
        if u == target: break
        for v, w in adj[u].items():
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd; prev[v] = u
                heapq.heappush(pq, (nd, v))
    if target not in dist:
        return math.inf, [], nodes_expanded
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return dist[target], path, nodes_expanded


def astar(adj, source, target, coords):
    def h(node):
        return haversine_km(coords[node][0], coords[node][1], coords[target][0], coords[target][1])
    g = {source: 0.0}; f = {source: h(source)}; prev = {}
    open_set = [(f[source], source)]; closed = set(); nodes_expanded = 0
    while open_set:
        _, u = heapq.heappop(open_set)
        if u in closed: continue
        closed.add(u); nodes_expanded += 1
        if u == target: break
        for v, w in adj[u].items():
            tg = g[u] + w
            if v not in g or tg < g[v]:
                g[v] = tg; f[v] = tg + h(v); prev[v] = u
                heapq.heappush(open_set, (f[v], v))
    if target not in g:
        return math.inf, [], nodes_expanded
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return g[target], path, nodes_expanded


def floyd_warshall(adj, n):
    INF = math.inf
    D = np.full((n, n), INF); np.fill_diagonal(D, 0.0)
    nxt = np.full((n, n), -1, dtype=int)
    for u in adj:
        for v, w in adj[u].items():
            D[u, v] = w; nxt[u, v] = v
    for i in range(n):
        nxt[i, i] = i
    for k_ in range(n):
        new_D = D[:, k_][:, None] + D[k_, :][None, :]
        mask = new_D < D
        if mask.any():
            rows, cols = np.where(mask)
            nxt[rows, cols] = nxt[rows, k_]
            D[mask] = new_D[mask]
    return D, nxt


def solve_route(points_df):
    """TSP sıralaması: N<=8 için tam (brute-force) arama, N>8 için en-yakın-komşu + 2-opt.
    §3.2.2, §3.2.3 ve §3.2.4'te değiştirilmeden yeniden kullanılmaktadır."""
    pts = points_df.reset_index(drop=True)
    n = len(pts)
    if n <= 1:
        return pts, 0.0
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = haversine_km(pts.lat[i], pts.lon[i], pts.lat[j], pts.lon[j])
    if n <= 8:
        best_order, best_total = None, math.inf
        for perm in permutations(range(n)):
            total = sum(D[perm[k], perm[k + 1]] for k in range(n - 1))
            if total < best_total:
                best_total, best_order = total, perm
    else:
        unvisited = set(range(n)); cur = 0
        order = [cur]; unvisited.remove(cur)
        while unvisited:
            nxt_ = min(unvisited, key=lambda j: D[cur, j])
            order.append(nxt_); unvisited.remove(nxt_); cur = nxt_
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    a, b, c, d = order[i-1], order[i], order[j], order[j+1]
                    if D[a, b] + D[c, d] > D[a, c] + D[b, d] + 1e-9:
                        order[i:j+1] = order[i:j+1][::-1]; improved = True
        best_order = order
        best_total = sum(D[best_order[k], best_order[k+1]] for k in range(n - 1))
    return pts.iloc[list(best_order)].reset_index(drop=True), best_total


if __name__ == '__main__':
    random.seed(42); np.random.seed(42)
    df = pd.read_csv('merged_dataset.csv')
    df = df[df['yapim_adi'].notnull()].copy()

    print("=== İstanbul grafı kuruluyor ===")
    adj, coords, names, locations = build_region_graph(df, 'İstanbul', k=6)
    n = len(locations)
    print(f"n={n} düğüm, {sum(len(v) for v in adj.values())//2} kenar "
          f"(beklenen: n=97, kenar=382)")
    assert n == 97, "Düğüm sayısı beklenenle uyuşmuyor!"

    # --- Karşılaştırmalı benchmark (Tablo 2) ---
    print("\n=== Dijkstra / A* / Floyd-Warshall karşılaştırması (30 rastgele çift) ===")
    D_fw, _ = floyd_warshall(adj, n)
    pairs = []
    while len(pairs) < 30:
        s, t = random.randint(0, n-1), random.randint(0, n-1)
        if s != t: pairs.append((s, t))

    d_nodes, a_nodes, matches = [], [], []
    for s, t in pairs:
        d_dist, _, d_ne = dijkstra(adj, s, t)
        a_dist, _, a_ne = astar(adj, s, t, coords)
        matches.append(abs(d_dist - D_fw[s, t]) < 1e-6 and abs(a_dist - D_fw[s, t]) < 1e-6)
        d_nodes.append(d_ne); a_nodes.append(a_ne)

    print(f"Doğruluk: {sum(matches)}/30 eşleşme (beklenen: 30/30)")
    print(f"Ort. ziyaret edilen düğüm - Dijkstra: {np.mean(d_nodes):.1f}, "
          f"A*: {np.mean(a_nodes):.1f} (beklenen: ~50.4 / ~16.0)")
    reduction = 100 * (1 - np.mean(a_nodes) / np.mean(d_nodes))
    print(f"A* düğüm azaltma oranı: %{reduction:.1f} (beklenen: ~%68.2)")
    assert sum(matches) == 30, "Doğruluk beklenenle uyuşmuyor!"

    # --- Örnek olay: Sadakatsiz (Balat -> Moda) ---
    print("\n=== Örnek olay: Sadakatsiz (Balat Sokakları -> Moda Sahili) ===")
    name_list = list(names)
    s_idx, t_idx = name_list.index('Balat Sokakları'), name_list.index('Moda Sahili')
    dist, path, _ = dijkstra(adj, s_idx, t_idx)
    print(f"Mesafe: {dist:.2f} km, Durak: {len(path)} (beklenen: 8.71 km, 6 durak)")
    for i in path:
        print(f"  - {names[i]}")
    assert abs(dist - 8.71) < 0.01 and len(path) == 6, "Sadakatsiz örneği beklenenle uyuşmuyor!"

    print("\n✓ Tüm doğrulamalar geçti.")
