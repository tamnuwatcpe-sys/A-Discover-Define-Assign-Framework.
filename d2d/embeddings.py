"""Sentence and word embeddings with multilingual-e5-large."""
import os

import numpy as np

from .config import EMBEDDING_MODEL

_model = None


def encode(texts, batch_size=128):
    """Unit-normalized e5 embeddings. The model is loaded once, on D2D_DEVICE (default: cuda)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL, device=os.environ.get("D2D_DEVICE", "cuda"))
    return _model.encode(["query: " + t for t in texts], normalize_embeddings=True, batch_size=batch_size)


def unit(M):
    """Scale every row to unit length."""
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def corpus_embeddings(corpus):
    """Embeddings of every text in the corpus, cached as emb.npy in the dataset's result folder."""
    path = corpus.spec.out_dir / "emb.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        np.save(path, encode(corpus.df.text.tolist()))
    return np.load(path)
