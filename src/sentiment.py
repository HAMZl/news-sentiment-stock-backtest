"""Headline sentiment scoring.

Two scorers are compared:

* **VADER** - a rule/lexicon model tuned for social-media text.  Fast, no GPU,
  but it was never trained on financial language ("shares fall", "beats
  estimates", "liability" ...).
* **FinBERT** (ProsusAI/finbert) - BERT fine-tuned on the Financial
  PhraseBank.  Score = P(positive) - P(negative), in [-1, 1].

Scores are cached to ``cache/`` because FinBERT takes several minutes on CPU.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "cache"


def _key(texts: pd.Series, name: str) -> Path:
    h = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:10]
    return CACHE / f"sentiment_{name}_{h}.csv"


def vader_scores(texts: pd.Series) -> np.ndarray:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(t)["compound"] for t in texts])


def finbert_scores(texts: pd.Series, batch_size: int = 64) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    pos, neg = labels["positive"], labels["negative"]
    torch.set_num_threads(max(1, torch.get_num_threads()))

    out = np.empty(len(texts))
    texts = list(texts)
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                      max_length=96, return_tensors="pt")
            p = model(**enc).logits.softmax(-1).numpy()
            out[i:i + batch_size] = p[:, pos] - p[:, neg]
            if (i // batch_size) % 50 == 0:
                print(f"  finbert {i:>6}/{len(texts)}", flush=True)
    return out


SCORERS = {"vader": vader_scores, "finbert": finbert_scores}


def score(texts: pd.Series, name: str) -> np.ndarray:
    """Score texts with the named model, using an on-disk cache."""
    CACHE.mkdir(exist_ok=True)
    path = _key(texts, name)
    if path.exists():
        return pd.read_csv(path)["score"].to_numpy()
    s = SCORERS[name](texts)
    pd.DataFrame({"score": s}).to_csv(path, index=False)
    return s
