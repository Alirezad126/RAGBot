"""The baked index: chunk texts as JSON, their vectors as a .npy array.

RAG-GPT indexes a document the visitor just uploaded, so its index has to live
in S3 between invocations. Here the corpus never changes, so the index is built
when the image is built and read from the image's own filesystem. There is no
bucket to pay for and no object to fetch before answering.

Both files load once per execution environment and are then held for the life
of that container, so the cost is a cold start, not a question.
"""

import json
from pathlib import Path

import numpy as np

from app.config import INDEX_DIR

_index = None


def load():
    """Return (chunks, vectors) for the baked corpus."""
    global _index
    if _index is None:
        directory = Path(INDEX_DIR)
        chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        vectors = np.load(directory / "vectors.npy", allow_pickle=False)

        if len(chunks) != len(vectors):
            raise ValueError(
                f"Index is inconsistent: {len(chunks)} chunks, {len(vectors)} vectors"
            )

        _index = (chunks, vectors)
    return _index
