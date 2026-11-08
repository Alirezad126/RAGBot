import numpy as np

from app.config import EMBEDDING_MODEL, TOP_K_PROFILE, TOP_K_THESIS

# Which sources count as biography. The thesis is everything else, and it is
# far larger, so the two groups are searched separately.
PROFILE_SOURCES = ("Profile notes", "CV")

_model = None


def model():
    """The ONNX embedding model, loaded once per execution environment."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(EMBEDDING_MODEL)
    return _model


def embed(texts):
    return np.asarray(list(model().embed(texts)), dtype=np.float32)


def _rank(query, vectors, rows):
    """`rows` ordered by how close their vectors are to `query`."""
    if not len(rows):
        return []
    subset = vectors[rows]

    # The model returns unit-length vectors, so a dot product is the cosine
    # similarity; normalize anyway so the ranking survives a model that doesn't.
    norms = np.linalg.norm(subset, axis=1) * np.linalg.norm(query)
    scores = subset @ query / np.where(norms == 0, 1, norms)

    return [rows[i] for i in np.argsort(scores)[::-1]]


def top_chunks(question, chunks, vectors):
    """The closest passages, with the CV and profile notes guaranteed slots.

    A plain top-k over the whole corpus is dominated by the thesis, which holds
    94% of the passages. Someone asking what Alireza does now would get eight
    paragraphs about compressed air storage and nothing from the CV. So the
    biography and the thesis are ranked separately and both are handed over,
    biography first, since that is what most visitors are actually asking about.
    """
    query = embed([question])[0]

    profile_rows = [
        i for i, chunk in enumerate(chunks) if chunk["source"] in PROFILE_SOURCES
    ]
    thesis_rows = [
        i for i, chunk in enumerate(chunks) if chunk["source"] not in PROFILE_SOURCES
    ]

    best = (
        _rank(query, vectors, profile_rows)[:TOP_K_PROFILE]
        + _rank(query, vectors, thesis_rows)[:TOP_K_THESIS]
    )
    return [chunks[i] for i in best]
