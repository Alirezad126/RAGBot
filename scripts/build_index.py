"""Build the corpus index. Runs during the image build, not at request time.

Reads corpus/, chunks it, embeds every chunk on CPU, and writes the two files
app/index_store.py expects. Because this happens at build time, a question
costs one query embedding and one Groq call, and nothing else.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.documents import load_chunks  # noqa: E402
from app.rag.retrieval import embed  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--out", default="index")
    args = parser.parse_args()

    chunks = load_chunks(args.corpus)
    print(f"Chunked the corpus into {len(chunks)} passages:")
    for source in dict.fromkeys(chunk["source"] for chunk in chunks):
        count = sum(1 for chunk in chunks if chunk["source"] == source)
        print(f"  {source}: {count}")

    vectors = embed([chunk["text"] for chunk in chunks])
    print(f"Embedded them as {vectors.shape[0]} x {vectors.shape[1]} float32.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    np.save(out / "vectors.npy", vectors)
    print(f"Wrote {out}/chunks.json and {out}/vectors.npy")


if __name__ == "__main__":
    main()
