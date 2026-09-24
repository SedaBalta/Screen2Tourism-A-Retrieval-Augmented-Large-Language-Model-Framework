"""
§3.2.4 LLM-BASED ROUTE GENERATION
====================================
f(R) = Narrate(Order(Select(Extract(R))))

Extract ve Narrate burada kural-tabanlı bir implementasyondur.
Select = §3.2.2'nin generate_hybrid_route(); Order = §3.2.1'in solve_route() - doğrudan import edilmektedir.

NOT: §3.2.4'ün doğrulama deneyi manuel olarak, harici bir LLM arayüzü üzerinden yürütülmüştür ve kod ile yeniden üretilebilir
değildir; bu dosya yalnızca f(R) boru hattının kendisini ve Kapadokya örnek olayını (Tablo 4'te bildirilen sayılarla) doğrulamaktadır. 
"""

import re
import numpy as np
import pandas as pd

import importlib.util


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sp = _load_module('02_shortest_path_method.py', 'sp_module')
solve_route = _sp.solve_route

_cb = _load_module('03_content_based_method.py', 'cb_module')
build_features = _cb.build_features
generate_hybrid_route = _cb.generate_hybrid_route
THEME_KEYWORDS = _cb.THEME_KEYWORDS


REGION_ALIASES = {
    'cappadocia': 'Nevşehir', 'kapadokya': 'Nevşehir', 'istanbul': 'İstanbul', 'izmir': 'İzmir',
}


def extract_intent(prompt):
    """Extract(R): serbest metni yapılandırılmış sorguya çevirir (kural-tabanlı yedek)."""
    p = prompt.lower()
    region = next((v for k, v in REGION_ALIASES.items() if k in p), None)
    themes = []
    if 'historical' in p or 'tarihi' in p: themes.append('tarihi')
    if 'cultural' in p or 'kültür' in p: themes.append('kültürel')
    m = re.search(r'(\d+)\s*day', p)
    days = int(m.group(1)) if m else 1
    return {'region': region, 'themes': themes or ['tarihi'], 'days': days}


def split_days(stops, n_days):
    if n_days <= 1 or len(stops) <= n_days:
        return [stops]
    idx_groups = np.array_split(np.arange(len(stops)), n_days)
    return [stops.iloc[idx].reset_index(drop=True) for idx in idx_groups]


def narrate(route):
    """Narrate(): yapılandırılmış rotayı doğal dil anlatımına çevirir (şablon-tabanlı yedek)."""
    lines = []
    for _, row in route.iterrows():
        act = row.get('populer_aktiviteler', '')
        lines.append(f"{row['lokasyon_adi']} durağında "
                     f"{act if pd.notnull(act) and act else 'bölgenin dokusunu'} keşfedin.")
    return " ".join(lines)


def generate_from_prompt(df, features, prompt):
    """f(R) = Narrate(Order(Select(Extract(R))))"""
    intent = extract_intent(prompt)
    selected, threshold = generate_hybrid_route(df, features, theme_interests=intent['themes'],
                                                 ref_region=intent['region'], geo_decay_km=20.0)
    route, total = solve_route(selected)
    days = split_days(route, intent['days'])
    return {
        'intent': intent, 'route': route, 'total_km': total,
        'threshold': threshold, 'days': days, 'narrative': narrate(route),
    }


if __name__ == '__main__':
    df = pd.read_csv('merged_dataset.csv')
    df = df[df['yapim_adi'].notnull()].copy()
    features = build_features(df)

    prompt = ("User likes historical dramas, has 2 days in Cappadocia and prefers "
              "cultural experiences.")
    print(f"=== Prompt: \"{prompt}\" ===\n")

    out = generate_from_prompt(df, features, prompt)
    print("Extract çıktısı:", out['intent'], "(beklenen bölge: Nevşehir, gün: 2)")
    print(f"Eşik: {out['threshold']:.4f} (beklenen: 0.1219)")
    print(f"Durak: {len(out['route'])} (beklenen: 7)")
    print(f"Mesafe: {out['total_km']:.2f} km (beklenen: 38.03 km)")
    print(f"Gün sayısı: {len(out['days'])} (beklenen: 2)")
    for i, day in enumerate(out['days']):
        print(f"  Gün {i+1}: {list(day['lokasyon_adi'])}")
    print("\nAnlatım (Narrate):\n", out['narrative'][:300], "...")

    assert out['intent']['region'] == 'Nevşehir' and out['intent']['days'] == 2
    assert len(out['route']) == 7
    assert abs(out['total_km'] - 38.03) < 0.05
    assert len(out['days']) == 2
    print("\n✓ Tüm doğrulamalar geçti.")


    print("\n" + "="*70)
    print("REFERANS - Harici LLM (ChatGPT) doğrulama deneyi sonuçları (Tablo 4):")
    print("="*70)
    ablation = pd.DataFrame([
        {'Koşul': 'A - yalnızca prompt', 'Örtüşme': '1/8'},
        {'Koşul': 'B - +veri, talimatsız', 'Örtüşme': '1/8 (A ile özdeş)'},
        {'Koşul': 'C - +"yalnızca bu veriyi kullan"', 'Örtüşme': '6/7 (Jaccard=0.667)'},
        {'Koşul': 'D - +"koordinatlara göre optimize et"', 'Örtüşme': '7/7 (Jaccard=0.875)'},
    ])
    print(ablation.to_string(index=False))
    print("\n(Bu tablo kod ile yeniden üretilemez; harici bir LLM arayüzü üzerinden "
          "manuel olarak yürütülmüştür - bkz. paper §3.2.4.4)")
