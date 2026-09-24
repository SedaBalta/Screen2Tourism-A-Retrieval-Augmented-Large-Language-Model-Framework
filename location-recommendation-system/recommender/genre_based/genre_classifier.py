import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline(class_weight="balanced", C=5.0):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=15000,
                                  sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=1000)),
    ])


class GenreClassifier:
    def __init__(self, pipeline=None, id2label=None):
        self.pipeline = pipeline or build_pipeline()
        self.id2label = id2label or {}

    def fit(self, texts, label_ids, id2label):
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.pipeline.fit(list(texts), np.asarray(label_ids))
        return self

    def predict_proba(self, text):
        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        return {self.id2label[int(c)]: float(p) for c, p in zip(classes, probs)}

    def predict(self, text, top_k=1):
        ranked = sorted(self.predict_proba(text).items(), key=lambda x: -x[1])
        return ranked[0][0] if top_k == 1 else [g for g, _ in ranked[:top_k]]

    def save(self, path):
        joblib.dump({"pipeline": self.pipeline, "id2label": self.id2label}, path)

    @classmethod
    def load(cls, path):
        obj = joblib.load(path)
        return cls(obj["pipeline"], obj["id2label"])
