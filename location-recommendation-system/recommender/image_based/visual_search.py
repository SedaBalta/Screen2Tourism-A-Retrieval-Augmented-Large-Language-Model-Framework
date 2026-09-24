from io import BytesIO

import numpy as np
import torch
from PIL import Image


def load_index(path):
    raw = np.load(path, allow_pickle=True).item()
    keys = list(raw.keys())
    matrix = np.asarray([raw[k] for k in keys], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    locations = np.array([int(k.split("_")[0]) for k in keys])
    return keys, matrix, locations


def rank_locations(similarities, locations, k):
    seen, ranked = set(), []
    for j in np.argsort(similarities)[::-1]:
        loc = int(locations[j])
        if loc not in seen:
            seen.add(loc)
            ranked.append((loc, float(similarities[j])))
        if len(ranked) >= k:
            break
    return ranked


class VisualLocationSearch:
    def __init__(self, index_path, device=None):
        import clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.keys, self.matrix, self.locations = load_index(index_path)

    def encode(self, image):
        if isinstance(image, (bytes, bytearray)):
            image = Image.open(BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)
        x = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v = self.model.encode_image(x).cpu().numpy().flatten().astype(np.float32)
        return v / np.linalg.norm(v)

    def search(self, image, k=5):
        return rank_locations(self.matrix @ self.encode(image), self.locations, k)
