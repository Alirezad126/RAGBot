import logging
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import APIStatusError
from pydantic import BaseModel

from app import index_store
from app.config import ALLOWED_ORIGINS, MAX_MESSAGE_CHARS
from app.rag.answer import answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAGBot",
    description="Ask Alireza's portfolio about his work.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Turn(BaseModel):
    message: str
    type: str
    id: Optional[int] = None
    loading: Union[bool, None] = None


class ChatRequest(BaseModel):
    message: str
    conversationState: List[Turn] = []


@app.get("/health")
async def health():
    chunks, _ = index_store.load()
    return {"status": "ok", "chunks": len(chunks)}


@app.post("/chat")
async def chat(body: ChatRequest):
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Ask something first.")
    if len(message) > MAX_MESSAGE_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Keep questions under {MAX_MESSAGE_CHARS} characters.",
        )

    chunks, vectors = index_store.load()
    try:
        response = answer(message, body.conversationState, chunks, vectors)
    except APIStatusError as exc:
        # Pass the provider's own status through rather than turning a rate
        # limit into an opaque 500.
        logger.warning("Provider error: %s", exc)
        if exc.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="Rate limited by the free tier, try again in a minute.",
            )
        raise HTTPException(
            status_code=502,
            detail=f"The model provider returned {exc.status_code}.",
        )
    return {"AIResponse": response}
