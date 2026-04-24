"""
Embedding module — wraps sentence-transformers for vibe text encoding.
Uses 'all-MiniLM-L6-v2' (dim=384, fast, good for semantic similarity).
"""

from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("Loading embedding model (first run only)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str) -> list:
    """Embed a single string → list of floats (384-dim)."""
    model = get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts: list) -> list:
    """Embed a batch of strings → list of vectors."""
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vecs]
