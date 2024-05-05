from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Union
from RAGModel.LLMModel import get_completion


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows only specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ConversationState(BaseModel):
    message: str
    type: str
    id: int
    loading: Union[bool, None] = None  # `None` allows for optional field

class RequestBody(BaseModel):
    message: str
    conversationState: List[ConversationState]

@app.post("/chat")
async def result(body: RequestBody):
    completion = get_completion(body)
    return {"AIResponse": completion}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info", reload=True)
