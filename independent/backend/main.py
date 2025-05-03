# backend/main.py
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, Dict
from typing_extensions import TypedDict
from in_p import answer_rag_json

BASE_DIR     = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

class PageReference(TypedDict):
    label: str
    page: int

class SymbolItem(TypedDict):
    name: str
    description: str
    analysis: str
    key_quote: str
    quote_page: int
    page_references: list[PageReference]

class AskResponse(TypedDict):
    title: str
    subtitle: str
    items: list[SymbolItem]

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    result: Dict[str,Any] = answer_rag_json(request.question)
    return result

app.mount(
    "/", 
    StaticFiles(directory=str(FRONTEND_DIR), html=True), 
    name="frontend"
)
