from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from in_p import answer_rag   

app = FastAPI()

# CORS to POST
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# endpoint
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    cards: list[str]

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    print("▶️ Received question:", req.question)
    ans = answer_rag(req.question)
    cards = [s.strip() + "." for s in ans.split(".") if s.strip()]
    return AskResponse(cards=cards)

# index.html at GET /
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("index.html")

# all other static files (CSS, JS) under /static
app.mount(
    "/static",
    StaticFiles(directory=".", html=False),
    name="static"
)
