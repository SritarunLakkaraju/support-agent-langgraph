"""Embedding models behind one interface.

* HashingEmbedder - offline, deterministic, no downloads. Word unigrams + bigrams + character
  trigrams are hashed into a fixed-size vector (the "hashing trick"), weighted by IDF when fitted,
  then L2-normalised so a dot product equals cosine similarity.
* VertexEmbedder  - Vertex AI text embeddings (text-embedding-005). Used when GOOGLE_CLOUD_PROJECT
  and FDE_USE_VERTEX=1 are set and google-genai is installed.

get_embedder() picks Vertex if configured, otherwise the offline embedder.
"""
from __future__ import annotations

import math
import os
import zlib
from collections import Counter
from typing import Protocol, Sequence

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _h(s: str) -> int:
    # stable across runs (the built-in hash() is randomly salted per process)
    return zlib.crc32(s.encode("utf-8"))


class HashingEmbedder:
    def __init__(self, dim: int = 768, char_ngrams: bool = True):
        self.dim = dim
        self.char_ngrams = char_ngrams
        self.idf: dict[str, float] = {}
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> Counter:
        toks = tokenize(text)
        feats = Counter(toks)
        feats.update(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
        if self.char_ngrams:
            for t in toks:
                padded = f"#{t}#"
                feats.update(f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2))
        return feats

    def fit(self, corpus: Sequence[str]) -> "HashingEmbedder":
        """Learn IDF weights so rare, informative features dominate the vector."""
        df = Counter()
        for doc in corpus:
            df.update(set(self._features(doc)))
        n = len(corpus)
        self.idf = {f: math.log((1 + n) / (1 + c)) + 1.0 for f, c in df.items()}
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for f, c in self._features(text).items():
                h = _h(f)
                sign = 1.0 if (h >> 31) & 1 else -1.0  # signed hashing reduces collision bias
                w = (1.0 + math.log(c)) * self.idf.get(f, 1.0)
                if f.startswith("c:"):
                    w *= 0.35  # sub-word features help with typos/morphology but should not dominate
                out[row, h % self.dim] += sign * w
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class VertexEmbedder:
    """Vertex AI embeddings via the google-genai SDK (run `gcloud auth application-default login` first)."""

    def __init__(self, model: str = "text-embedding-005", dim: int = 768):
        from google import genai  # optional dependency, imported lazily

        self.client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                                   location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
        self.model, self.dim, self.name = model, dim, f"vertex:{model}"

    def fit(self, corpus):  # API-compatible no-op
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), 100):  # API batch limit
            resp = self.client.models.embed_content(model=self.model, contents=list(texts[i:i + 100]))
            vecs.extend(e.values for e in resp.embeddings)
        arr = np.asarray(vecs, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def get_embedder(dim: int = 768):
    if os.environ.get("GOOGLE_CLOUD_PROJECT") and os.environ.get("FDE_USE_VERTEX") == "1":
        return VertexEmbedder(dim=dim)
    return HashingEmbedder(dim=dim)
