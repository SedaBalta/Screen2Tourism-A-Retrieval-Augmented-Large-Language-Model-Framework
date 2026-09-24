import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def download(csv_path, images_dir, timeout=20):
    import requests

    images_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    ok = 0
    for _, r in df.iterrows():
        dest = images_dir / f"{int(r['lokasyon_id'])}_{int(r['gorsel_id'])}.jpg"
        if dest.exists():
            ok += 1
            continue
        try:
            resp = requests.get(str(r["gorsel_url"]), timeout=timeout,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            ok += 1
        except Exception:
            pass
    print(f"downloaded {ok} of {len(df)} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default=str(ROOT / "data/visual/images"))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--csv", default=str(ROOT / "data/catalogue/lokasyon_gorselleri.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/visual/clip_index.npy"))
    a = ap.parse_args()

    import clip

    images_dir = Path(a.images_dir)
    if a.download:
        download(a.csv, images_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    index = {}
    for path in sorted(images_dir.iterdir()):
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        except Exception:
            continue
        with torch.no_grad():
            index[path.stem] = model.encode_image(image).cpu().numpy().flatten().astype(np.float32)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, index, allow_pickle=True)
    print(f"encoded {len(index)} images -> {a.out}")


if __name__ == "__main__":
    main()
