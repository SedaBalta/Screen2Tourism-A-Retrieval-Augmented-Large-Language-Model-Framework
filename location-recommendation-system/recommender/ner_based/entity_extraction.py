import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

DEFAULT_ID2LABEL = {0: "O", 1: "B-MEKAN", 2: "I-MEKAN", 3: "B-ZAMAN",
                    4: "I-ZAMAN", 5: "B-OLAY", 6: "I-OLAY"}

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


class EntityExtractor:
    def __init__(self, model_dir, max_length=128, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.max_length = max_length
        cfg = self.model.config.id2label
        self.id2label = ({int(k): v for k, v in cfg.items()}
                         if cfg and not str(cfg.get(0, "")).startswith("LABEL_")
                         else DEFAULT_ID2LABEL)

    def spans(self, text):
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=self.max_length).to(self.device)
        with torch.no_grad():
            pred = torch.argmax(self.model(**enc).logits, dim=2)[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        out, current, current_type = [], [], None
        for tok, p in zip(tokens, pred):
            if tok in SPECIAL_TOKENS:
                continue
            label = self.id2label[p]
            if label != "O":
                piece = tok.replace("##", "")
                if tok.startswith("##") and current:
                    current[-1] += piece
                else:
                    current.append(piece)
                if current_type is None:
                    current_type = label.split("-")[-1]
            elif current:
                out.append((current_type, " ".join(current)))
                current, current_type = [], None
        if current:
            out.append((current_type, " ".join(current)))
        return out

    def extract(self, text):
        return " ".join(s for _, s in self.spans(text))
