from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qa_rag import answer_rag

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            
    allow_methods=["POST"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    cards: list[str]

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    # take rag response
    answer = answer_rag(req.question)
    # return flush card
    return AskResponse(cards=[answer])