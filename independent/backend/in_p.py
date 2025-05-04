import os
os.environ["TOKENIZERS_PARALLELISM"]    = "true"
os.environ["OMP_NUM_THREADS"]          = "8"
os.environ["MKL_NUM_THREADS"]          = "8"
os.environ["NUMEXPR_NUM_THREADS"]      = "8"
os.environ["TRANSFORMERS_NO_TF"]       = "1"   

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⠂ Running on device = {DEVICE}")

import pickle
import json
import logging
import faiss
import openai
import torch
import time
import re
from fastapi import HTTPException
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, CrossEncoder


# secret API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")

openai.api_key = api_key

# Paths
BASE_DIR = "../"
DATA_DIR = os.path.join(BASE_DIR, "data")   # <project root>/data

PDF_PATH    = os.path.join(DATA_DIR, "the-great-gatsby.pdf")
INDEX_PATH  = os.path.join(DATA_DIR, "gatsby_index.faiss")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")

logger = logging.getLogger(__name__)

def load_pdf(path: str) -> str:
    return extract_text(path)

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        chunks.append(" ".join(tokens[start : start + chunk_size]))
        start += chunk_size - overlap
    return chunks

def build_index(chunks: list[str], model_name: str = "paraphrase-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name, device=DEVICE)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Build or load FAISS index + chunks
if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
    text = load_pdf(PDF_PATH)
    chunks = chunk_text(text)
    index = build_index(chunks)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks and FAISS index.")
else:
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

# Embedding and cross-encoder models
embed_model   = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=DEVICE)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE)

def retrieve(question: str, top_k: int = 5, fetch_k: int = 3) -> list[str]:
    q_vec = embed_model.encode(question, convert_to_numpy=True)
    _, ids = index.search(q_vec.reshape(1, -1), fetch_k)
    candidates = [chunks[i] for i in ids[0]]
    pairs = [(question, txt) for txt in candidates]  
    # t0 = time.perf_counter()
    scores = cross_encoder.predict(pairs, batch_size=len(pairs))
    # t1 = time.perf_counter() - t0
    # print(f"re-rank (batched) took {t1:.3f}s")
    ranked = [txt for _, txt in sorted(zip(scores, candidates), reverse=True)]
    return ranked[:top_k]

def answer_rag_json(question: str, top_k: int = 3) -> dict:
    docs = retrieve(question, top_k=top_k)
    context = "\n\n".join(d.replace("\n", " ") for d in docs)

    system_prompt = (
        "You are a knowledgeable literary analysis assistant for the book the Great Gatsby. "
        "For each user question, output only a single JSON object exactly matching the schema."
        "Always include a `quote_page` integer for each symbol and the full, un-truncated `key_quote` end with period. "
        "Supply a `page_references` array with as many descriptive entries as are relevant—do not default to any particular number. "
        "For the `analysis` field, provide a concise yet thoughtful multi-sentence explanation (around 2-3 sentences) that elaborates on the symbol's thematic role without imposing a strict word limit. "
        "Every field name and string must be double-quoted."
        "In page_references, each object must have exactly two properties: Do not emit any bare numbers or extra fields."
    )

    few_shot_messages = [
        {
            "role": "user",
            "content": """
Context (from relevant passages):
“He stretched out his arms toward the dark water in a curious way and far as I was from him I could have sworn he was trembling. Involuntarily I glanced seaward—and distinguished nothing except a single green light, minute and far away, that might have been the end of a dock.”

User's question:
"What does the green light symbolize?"

Please answer using the JSON schema:
{
  "title": "<Your title>",
  "subtitle": "<Your subtitle>",
  "items": [ { /* one symbol */ } ]
}
"""
        },
        {
            "role": "assistant",
            "content": """
{
  "title": "Symbolism of the Green Light",
  "subtitle": "An exploration of Gatsby's hopes and the broader American Dream.",
  "items": [
    {
      "name": "The Green Light",
      "description": "A beacon at the end of Daisy's dock symbolizing hope and aspiration.",
      "analysis": "The green light represents Gatsby's yearning for Daisy and the promise of a newer future. It also embodies the larger American Dream, calling attention to both its allure and its inevitable elusiveness as characters grapple with idealism versus reality.",
      "key_quote": "He stretched out his arms toward the dark water in a curious way and far as I was from him I could have sworn he was trembling. Involuntarily I glanced seaward—and distinguished nothing except a single green light, minute and far away, that might have been the end of a dock.",
      "quote_page": 23,
      "page_references": [
        {"label": "Initial mention on Daisy's dock", "page": 23},
        {"label": "Gatsby's reflective gaze", "page": 91}
      ]
    }
  ]
}
"""
        },
        {
            "role": "user",
            "content": """
Context (from relevant passages):
“This is a valley of ashes—a fantastic farm where ashes grow like wheat into ridges and hills and grotesque gardens; where ashes take the forms of houses...”

User's question:
"What is the significance of the Valley of Ashes?"

Please answer using the same JSON schema as above.
"""
        },
        {
            "role": "assistant",
            "content": """
{
  "title": "Meaning of the Valley of Ashes",
  "subtitle": "An analysis of social decay and moral corruption symbolized by the wasteland.",
  "items": [
    {
      "name": "The Valley of Ashes",
      "description": "A bleak wasteland between West Egg and New York City symbolizing industrial and moral decay.",
      "analysis": "The Valley of Ashes starkly depicts the fallout of unbridled ambition and materialism. Its dusty expanse reflects the moral emptiness beneath the era's glamorous facade, emphasizing the chasm between wealth and ethical integrity.",
      "key_quote": "This is a valley of ashes—a fantastic farm where ashes grow like wheat into ridges and hills and grotesque gardens; where ashes take the forms of houses...",
      "quote_page": 27,
      "page_references": [
        {"label": "Eckleburg's eyes overlooking the ashes", "page": 27},
        {"label": "Scene of Myrtle's demise", "page": 156},
        {"label": "Nick's reflection on its symbolism", "page": 162}
      ]
    }
  ]
}
"""
        }
    ]

    schema = """{
  "title": "<A concise, question-derived title>",
  "subtitle": "<A descriptive subtitle reflecting the question's focus>",
  "items": [
    {
      "name": "<Symbol or motif name>",
      "description": "<Brief description>",
      "analysis": "<An explanation of around 2-3 sentences that explores thematic significance>",
      "key_quote": "<The full, un-truncated quote illustrating the symbol>",
      "quote_page": <int>,
      "page_references": [
        { "label": "<Descriptive label>", "page": <int> }
      ]
    }
  ]
}"""

    user_prompt = (
        f"Context (from relevant passages):\n{context}\n\n"
        f"User's question:\n\"{question}\"\n\n"
        "Please answer using the JSON schema below, generating one title and one subtitle based on the question. \n"
        "if you can generate 3 or more page references it will be great!. \n"
        "at least 4 items will be perfect!\n"
        + schema
    )

    messages = [{"role": "system", "content": system_prompt}] + few_shot_messages + [{"role": "user", "content": user_prompt}]
    t2 = time.perf_counter()
    try:
      response = openai.chat.completions.create(
          model="gpt-4.1-nano",
          messages=messages,
          temperature=0.5,
          timeout=10
      )
    except Exception as e:
        # detect timeouts:
        name = type(e).__name__
        msg  = str(e).lower()
        if name == "Timeout" or "timeout" in msg:
            # 504 Gateway Timeout
            raise HTTPException(status_code=504, detail="Upstream API request timed out")
        # otherwise re‑raise as a 502 Bad Gateway
        raise HTTPException(status_code=502, detail=f"Upstream API error: {e}")
    t3 = time.perf_counter() - t2
    print(f"[OpenAI] call took {t3:.3f}s")
    raw = response.choices[0].message.content.strip()
    # logger.debug("ChatGPT raw output: %s", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # If fails, fall back to grabbing the first {...} block
        match = re.search(r'(\{.*\})', raw, re.S)
        if match:
            candidate = match.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e2:
                logger.error("Fallback JSON decode failed: %s", e2)
                logger.error("Fallback candidate was: %s", candidate)
                raise
        else:
            logger.error("Could not find any JSON object in the model output")
            raise ValueError("Invalid JSON from ChatGPT—no object bracketing found")

# import time

# test_questions = [
#     # "What is the significance of the green light at the end of Daisy’s dock?",
#     # "How does the Valley of Ashes illustrate the novel’s critique of moral decay?",
#     # "What role does Nick Carraway play as narrator?",
#     # "In what ways does Daisy Buchanan embody both illusion and reality?",
#     # "Why does Gatsby throw lavish parties?",
#     # "How does The Great Gatsby portray different social classes",
#     "What are the major symbols in The Great Gatsby",
#     "What are the major themes of the novel",
      
# ]

# for q in test_questions:
#     start = time.perf_counter()
#     try:
#         answer = answer_rag_json(q)
#     except Exception as e:
#         answer = f"Error: {e}"
#     elapsed = time.perf_counter() - start
#     print("Q:", q)
#     print("A:", answer)
#     print(f"→ Time: {elapsed:.3f} s")
#     print("-" * 80)
