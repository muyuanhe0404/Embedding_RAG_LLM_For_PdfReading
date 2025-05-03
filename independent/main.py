from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict
from typing_extensions import TypedDict
from in_p import answer_rag_json

app = FastAPI()

# Allow your front-end origin (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

# Define the shape of the JSON you expect to return
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
    """
    Receives { question }, runs RAG + LLM, and returns the full
    analysis JSON, including subtitle, full quotes, quote_page,
    and page_references.
    """
    # Call into your updated in_p.answer_rag_json, which now
    # builds and parses the JSON schema we defined.
    result: Dict[str, Any] = answer_rag_json(request.question)
    return result

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("index.html")

app.mount("/static", StaticFiles(directory="."), name="static")
