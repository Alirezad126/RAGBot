"""Reading the corpus into labelled chunks.

Each chunk keeps the document it came from, so the prompt can tell the model
whether a passage is a CV line, a paragraph of the thesis, or a profile note.
Without that the model cites the thesis for things the CV says, and a reader
asking what Alireza does now gets an answer about compressed air storage.
"""

from pathlib import Path

from pypdf import PdfReader

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.rag.chunking import split_text

# Label, filename. The label appears verbatim in the prompt.
SOURCES = [
    ("Profile notes", "profile.md"),
    ("CV", "cv.pdf"),
    ("M.A.Sc. thesis", "thesis.pdf"),
]


def _read(path):
    if path.suffix.lower() != ".pdf":
        return path.read_text(encoding="utf-8")

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_chunks(corpus_dir):
    """Every chunk of the corpus as {"source": label, "text": chunk}."""
    corpus_dir = Path(corpus_dir)
    chunks = []

    for label, filename in SOURCES:
        path = corpus_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Corpus is missing {path}")

        text = _read(path)
        if not text.strip():
            raise ValueError(f"No extractable text in {path}")

        chunks.extend(
            {"source": label, "text": piece}
            for piece in split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        )

    return chunks
