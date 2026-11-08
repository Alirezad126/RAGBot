"""Recursive character splitting, without pulling in LangChain for it.

Text is broken on the coarsest separator that keeps pieces under the limit —
paragraphs before lines before sentences before words — then adjacent pieces
are merged back up to `size` with a trailing overlap, so a sentence spanning a
boundary still appears whole in one of the chunks.
"""

SEPARATORS = ["\n\n", "\n", ". ", ", ", " "]


def _atomize(text, size, separators=SEPARATORS):
    if len(text) <= size:
        return [text] if text.strip() else []

    for index, separator in enumerate(separators):
        if separator in text:
            pieces = []
            for piece in text.split(separator):
                pieces.extend(_atomize(piece, size, separators[index + 1 :]))
            return pieces

    # No separator left: a single unbroken run, so slice it.
    return [text[at : at + size] for at in range(0, len(text), size)]


def split_text(text, size, overlap):
    """Split `text` into chunks of at most `size` characters, overlapping."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks = []
    current = []  # pieces in the chunk being built

    def length(pieces):
        return sum(len(piece) for piece in pieces) + max(len(pieces) - 1, 0)

    for piece in _atomize(text, size):
        if current and length(current) + 1 + len(piece) > size:
            chunks.append(" ".join(current))
            # Carry whole trailing pieces, never a sliced word, up to `overlap`.
            while current and length(current) > overlap:
                current.pop(0)
            # The carried tail plus this piece may still not fit; drop the tail.
            if current and length(current) + 1 + len(piece) > size:
                current = []
        current.append(piece)

    if current:
        chunks.append(" ".join(current))
    return chunks
