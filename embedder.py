"""
Embedders.

An embedder turns text into a fixed-length vector of numbers. Two texts that mean
similar things should land close together in that number-space. Everything in Step 1
depends on this one assumption -- and Step 1 exists partly to show you where the
assumption breaks.

Three backends, same interface, so you can swap them without touching the store:

  TfidfEmbedder    - runs offline, no API key, no download. Word-overlap only.
                     Use this to get the mechanics working today.
  LocalEmbedder    - sentence-transformers, runs on your machine, real semantics.
                     Needs `pip install sentence-transformers` and a one-time download.
  VoyageEmbedder   - hosted API, best quality, needs a key.

Start on Tfidf. Move to LocalEmbedder as soon as you can install it -- and note how
the failure cases in step1 change (and which ones DON'T).
"""

from __future__ import annotations

import numpy as np


class BaseEmbedder:
    name = "base"

    def fit(self, corpus: list[str]) -> None:
        """Optional. Only fitted embedders (TF-IDF) need to see the corpus first."""
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbedder(BaseEmbedder):
    """
    Offline fallback. Scores a word higher when it is frequent in this document but
    rare across the whole collection. Purely lexical: it has no idea that "editor"
    and "Neovim" are related unless those words co-occur.

    Its blind spot is deliberate and instructive.
    """

    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
        )
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        self._vec.fit(corpus)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit(corpus) before encode() on TfidfEmbedder.")
        return np.asarray(self._vec.transform(texts).todense(), dtype=np.float32)


class LocalEmbedder(BaseEmbedder):
    """Real semantic embeddings, running locally. Preferred once installed."""

    name = "sentence-transformers"

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts), dtype=np.float32)


class VoyageEmbedder(BaseEmbedder):
    """Hosted embeddings. Anthropic's documented recommendation for embedding work."""

    name = "voyage"

    def __init__(self, model: str = "voyage-3", api_key: str | None = None):
        import os

        import voyageai

        self._client = voyageai.Client(api_key=api_key or os.environ["VOYAGE_API_KEY"])
        self._model = model

    def encode(self, texts: list[str]) -> np.ndarray:
        out = self._client.embed(texts, model=self._model)
        return np.asarray(out.embeddings, dtype=np.float32)


def get_embedder(prefer: str = "auto") -> BaseEmbedder:
    """Pick the best embedder actually available in this environment."""
    if prefer in ("auto", "local"):
        try:
            return LocalEmbedder()
        except Exception:
            if prefer == "local":
                raise
    if prefer in ("auto", "voyage"):
        try:
            return VoyageEmbedder()
        except Exception:
            if prefer == "voyage":
                raise
    return TfidfEmbedder()
