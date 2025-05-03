import os
os.environ["TOKENIZERS_PARALLELISM"]    = "false"
os.environ["OMP_NUM_THREADS"]          = "1"
os.environ["MKL_NUM_THREADS"]          = "1"
os.environ["NUMEXPR_NUM_THREADS"]      = "1"
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

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
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

def retrieve(question: str, top_k: int = 3, fetch_k: int = 6) -> list[str]:
    q_vec = embed_model.encode(question, convert_to_numpy=True)
    _, ids = index.search(q_vec.reshape(1, -1), fetch_k)
    candidates = [chunks[i] for i in ids[0]]
    scores = cross_encoder.predict([(question, txt) for txt in candidates])
    ranked = [txt for _, txt in sorted(zip(scores, candidates), reverse=True)]
    return ranked[:top_k]

def answer_rag_json(question: str, top_k: int = 5) -> dict:
    docs = retrieve(question, top_k=top_k)
    context = "\n\n".join(d.replace("\n", " ") for d in docs)

    system_prompt = (
        "You are a knowledgeable literary analysis assistant. "
        "For each user question, first generate one concise `title` derived from the question, then one descriptive `subtitle`. "
        "Next, using the provided context, identify the core symbols or motifs present. "
        "Always include a `quote_page` integer for each symbol and the full, un-truncated `key_quote` end with period. "
        "Supply a `page_references` array with as many descriptive entries as are relevant—do not default to any particular number. "
        "For the `analysis` field, provide a concise yet thoughtful multi-sentence explanation (around 2–3 sentences) that elaborates on the symbol’s thematic role without imposing a strict word limit. "
        "For the page reference, give me at least 3 references depending on the questions."
        "Output only a single JSON object exactly matching the schema."
    )

    few_shot_messages = [
        {
            "role": "user",
            "content": """
Context (from relevant passages):
“He stretched out his arms toward the dark water in a curious way and far as I was from him I could have sworn he was trembling. Involuntarily I glanced seaward—and distinguished nothing except a single green light, minute and far away, that might have been the end of a dock.”

User’s question:
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
  "subtitle": "An exploration of Gatsby’s hopes and the broader American Dream.",
  "items": [
    {
      "name": "The Green Light",
      "description": "A beacon at the end of Daisy’s dock symbolizing hope and aspiration.",
      "analysis": "The green light represents Gatsby’s yearning for Daisy and the promise of a newer future. It also embodies the larger American Dream, calling attention to both its allure and its inevitable elusiveness as characters grapple with idealism versus reality.",
      "key_quote": "He stretched out his arms toward the dark water in a curious way and far as I was from him I could have sworn he was trembling. Involuntarily I glanced seaward—and distinguished nothing except a single green light, minute and far away, that might have been the end of a dock.",
      "quote_page": 23,
      "page_references": [
        {"label": "Initial mention on Daisy’s dock", "page": 23},
        {"label": "Gatsby’s reflective gaze", "page": 91}
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

User’s question:
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
      "analysis": "The Valley of Ashes starkly depicts the fallout of unbridled ambition and materialism. Its dusty expanse reflects the moral emptiness beneath the era’s glamorous facade, emphasizing the chasm between wealth and ethical integrity.",
      "key_quote": "This is a valley of ashes—a fantastic farm where ashes grow like wheat into ridges and hills and grotesque gardens; where ashes take the forms of houses...",
      "quote_page": 27,
      "page_references": [
        {"label": "Eckleburg’s eyes overlooking the ashes", "page": 27},
        {"label": "Scene of Myrtle’s demise", "page": 156},
        {"label": "Nick’s reflection on its symbolism", "page": 162}
      ]
    }
  ]
}
"""
        }
    ]

    schema = """{
  "title": "<A concise, question-derived title>",
  "subtitle": "<A descriptive subtitle reflecting the question’s focus>",
  "items": [
    {
      "name": "<Symbol or motif name>",
      "description": "<Brief description>",
      "analysis": "<An explanation of around 2–3 sentences that explores thematic significance>",
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
        f"User’s question:\n\"{question}\"\n\n"
        "Please answer using the JSON schema below, generating one title and one subtitle based on the question. \n"
        "3 page references or more will be great!. \n"
        "Also round 5 items will be perfect!\n"
        + schema
    )

    messages = [{"role": "system", "content": system_prompt}] + few_shot_messages + [{"role": "user", "content": user_prompt}]

    response = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()
    logger.debug("ChatGPT raw output: %s", raw)

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0 or start >= end:
        logger.error("Invalid JSON in response: %s", raw)
        raise ValueError("Invalid JSON from ChatGPT")

    json_str = raw[start : end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("JSON decode failed: %s", e)
        logger.error("JSON string was: %s", json_str)
        raise


    return data

# import time

# test_questions = [
#     # "What is the significance of the green light at the end of Daisy’s dock?",
#     # "How does the Valley of Ashes illustrate the novel’s critique of moral decay?",
#     # "What role does Nick Carraway play as narrator?",
#     # "In what ways does Daisy Buchanan embody both illusion and reality?",
#     # "Why does Gatsby throw lavish parties?",
#     # "How does The Great Gatsby portray different social classes",
#     "What are the major symbols in The Great Gatsby",
#     "What are the major themes of the novel"
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

