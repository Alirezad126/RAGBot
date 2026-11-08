import os

# Generation. Groq serves open-weight models on an OpenAI-compatible API and
# its free tier covers portfolio traffic, so answering a visitor costs nothing.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# One model, not a picker: a visitor clicking a chat bubble wants an answer,
# not a choice. The 20B is quick to first token and still good enough to stay
# inside the excerpts, which is the only thing being asked of it here.
MODEL = os.environ.get("MODEL", "openai/gpt-oss-20b")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "openai/gpt-oss-120b")

# The GPT-OSS models spend part of this budget on reasoning tokens before they
# emit anything, so a ceiling sized for three sentences cuts the answer off
# mid-word. Keep the effort low and the ceiling well clear of the prose.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 1024))
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "low")

# Embeddings run inside the function on CPU: an ONNX model baked into the
# image, so retrieval calls no API and adds nothing per question.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Retrieval. The thesis is 514 of the 546 passages, so a single top-k over the
# whole corpus lets it crowd out the CV and the profile notes on exactly the
# questions a visitor asks first. Each group gets its own slots instead.
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 64))
TOP_K_PROFILE = int(os.environ.get("TOP_K_PROFILE", 8))
TOP_K_THESIS = int(os.environ.get("TOP_K_THESIS", 4))

# The corpus is fixed, so unlike RAG-GPT there is no upload and no per-session
# state: the index is built when the image is built and read from local disk.
# That removes the S3 bucket, its requests, and a round trip per question.
INDEX_DIR = os.environ.get("INDEX_DIR", "/var/task/index")

# A short block of current facts put in front of every answer, whatever
# retrieval returned. "What are you working on now" is the most common question
# and the worst one to answer from whichever passage happened to score highest,
# so it does not go through retrieval at all.
PINNED_PATH = os.environ.get("PINNED_PATH", "/var/task/corpus/now.md")

# A single question from a chat bubble; anything longer is not a question.
MAX_MESSAGE_CHARS = int(os.environ.get("MAX_MESSAGE_CHARS", 600))
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", 8))

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:5173"
    ).split(",")
    if origin.strip()
]
