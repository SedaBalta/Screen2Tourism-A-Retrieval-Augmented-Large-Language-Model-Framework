"""
§3.1 DATASET CONSTRUCTION
==========================
Üç ham CSV dosyasını (yapimlar, yapim_lokasyon, lokasyonlar) birleştirdikten sonra merged_dataset.csv dosyasını üretir.

Girdi: yapimlar_temizlenmis__2___1_.csv, yapim_lokasyon_final__2_.csv, lokasyonlar_birlesik.csv
Çıktı: merged_dataset.csv (1811 satır, 283 benzersiz lokasyon)
"""

import pandas as pd


def parse_coord(s):
    lat, lon = s.split(',')
    return float(lat.strip()), float(lon.strip())


def build_dataset(yapim_path, link_path, lokasyon_path, output_path='merged_dataset.csv'):
    yapim = pd.read_csv(yapim_path)
    link = pd.read_csv(link_path)
    lok = pd.read_csv(lokasyon_path)

    # --- Veri kalitesi düzeltmesi: Moda Sahili koordinat hatası ---
    # Orijinal: 41.0868, 29.0472 (gerçek Moda'dan ~10km kuzeyde, Bebek'e yakın)
    # Düzeltilmiş: 40.9800, 29.0228 (bağımsız coğrafi referansla doğrulanmış)
    mask = lok['lokasyon_adi'] == 'Moda Sahili'
    if mask.any():
        lok.loc[mask, 'koordinat'] = '40.9800, 29.0228'

    lok[['lat', 'lon']] = lok['koordinat'].apply(lambda s: pd.Series(parse_coord(s)))
    lok_renamed = lok.rename(columns={'id': 'lokasyon_id'})

    # --- Birleştirme: köprü tablo + yapımlar (LEFT JOIN) + lokasyonlar (INNER JOIN) ---
    merged = link.merge(yapim, on='yapim_id', how='left')
    merged = merged.merge(lok_renamed, on='lokasyon_id', how='inner')

    cols = ['yapim_id', 'yapim_adi', 'yapim_turu', 'tur', 'yonetmen', 'yapimci',
            'yapim_yili', 'produksiyon_sirketi', 'oyuncular',
            'lokasyon_id', 'lokasyon_adi', 'sehir', 'ilce', 'lat', 'lon', 'koordinat',
            'turizm_potansiyeli', 'ziyaret_durumu', 'populer_aktiviteler', 'gorsel_url',
            'sahne_bilgisi']
    merged = merged[cols]
    merged.to_csv(output_path, index=False)
    return merged


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Veri seti birleştirme (§3.1)")
    ap.add_argument('--yapim', default='yapimlar_temizlenmis__2___1_.csv')
    ap.add_argument('--link', default='yapim_lokasyon_final__2_.csv')
    ap.add_argument('--lokasyon', default='lokasyonlar_birlesik.csv')
    ap.add_argument('--output', default='merged_dataset.csv')
    args = ap.parse_args()

    df = build_dataset(args.yapim, args.link, args.lokasyon, args.output)

    # --- Doğrulama: paper'da bildirilen sayılarla örtüşme kontrolü ---
    print(f"Toplam satır: {len(df)} (beklenen: 1811)")
    print(f"Benzersiz lokasyon: {df['lokasyon_id'].nunique()} (beklenen: 283-304 arası, "
          f"filtreye bağlı)")
    print(f"Yapım bilgisi eksik satır: {df['yapim_adi'].isnull().sum()} (beklenen: 59)")

    moda = df[df['lokasyon_adi'] == 'Moda Sahili'][['lat', 'lon']].drop_duplicates()
    print(f"\nModa Sahili düzeltilmiş koordinat: {moda.values.tolist()} "
          f"(beklenen: [[40.98, 29.0228]])")

    assert len(df) == 1811, "Satır sayısı beklenenle uyuşmuyor!"
    assert df['yapim_adi'].isnull().sum() == 59, "Eksik yapım sayısı beklenenle uyuşmuyor!"
    assert abs(moda.iloc[0]['lat'] - 40.98) < 1e-6, "Moda koordinat düzeltmesi uygulanmamış!"
    print("\n✓ Tüm doğrulamalar geçti.")
