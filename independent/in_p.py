import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
DATA_DIR = '/content/drive/MyDrive/independent'
os.makedirs(DATA_DIR, exist_ok=True)
PDF_PATH = os.path.join(DATA_DIR, 'the-great-gatsby.pdf')

import pickle
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import faiss

def load_pdf(path):
    return extract_text(path)

def chunk_text(text, chunk_size=500, overlap=100):
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        chunks.append(" ".join(tokens[start:start+chunk_size]))
        start += chunk_size - overlap
    return chunks

def build_index(chunks, model_name='paraphrase-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name, device='cuda')
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    idx = faiss.IndexFlatL2(embeddings.shape[1])
    idx.add(embeddings)
    return idx

text = load_pdf(PDF_PATH)
chunks = chunk_text(text)
index = build_index(chunks)
faiss.write_index(index, os.path.join(DATA_DIR, 'gatsby_index.faiss'))
with open(os.path.join(DATA_DIR, 'chunks.pkl'), 'wb') as f:
    pickle.dump(chunks, f)
print(f"Saved {len(chunks)} chunks and FAISS index.")


apikey = '<hugging face key>' #put secret key here
from huggingface_hub import login
login(token=apikey)

import torch
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=quant_config,
    device_map={"": "cuda:0"}     
)


tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    use_fast=False
)

generator = pipeline(
    "text-generation",      
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=60,     
    do_sample=True,         
    top_k=50,
    top_p=0.9,
    temperature=0.7,
    repetition_penalty=1.1,
    return_full_text=False
)



import faiss
import pickle
from sentence_transformers import SentenceTransformer,CrossEncoder

index = faiss.read_index(os.path.join(DATA_DIR, 'gatsby_index.faiss'))
with open(os.path.join(DATA_DIR, 'chunks.pkl'), 'rb') as f:
    chunks = pickle.load(f)

embed_model   = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

def retrieve(question: str, top_k: int = 3, fetch_k: int = 6):
    q_vec = embed_model.encode(question, convert_to_numpy=True)
    _, idxs = index.search(q_vec.reshape(1, -1), fetch_k)
    candidates = [chunks[i] for i in idxs[0]]
    scores = cross_encoder.predict([(question, txt) for txt in candidates])
    ranked = [txt for _, txt in sorted(zip(scores, candidates), reverse=True)]
    return ranked[:top_k]

def answer_rag(question: str, top_k: int = 5):
    docs = retrieve(question, top_k=top_k)
    context = "\n\n".join(d.replace("\n", " ") for d in docs)

    few_shot = """
Example:
Q: What does the Valley of Ashes symbolize in The Great Gatsby?
A: The Valley of Ashes symbolizes the moral decay lurking beneath the era’s glamour, representing the cost of unbridled industrial progress.

Example:
Q: How does Gatsby’s smile affect those around him?
A: Gatsby’s smile conveys both warmth and mystery, making others feel instantly at ease while also curious about the man behind it.

Now it’s your turn, answer in **exactly one sentence around 20 words** (no more, no less), ending with a single period:
"""

    prompt = (
        few_shot +
        "Context:\n" + context + "\n\n" +
        f"Q: {question}\n" +
        "A:"
    )
    raw = generator(prompt)[0]['generated_text']

    if "Example:" in raw:
        raw = raw.split("Example:")[0]

    answer = raw.strip().split("\n")[0]

    return answer.strip()

import time

test_questions = [
    "What is the significance of the green light at the end of Daisy’s dock?",
    "How does the Valley of Ashes illustrate the novel’s critique of moral decay?",
    "What role does Nick Carraway play as narrator, and how does his perspective shape the story?",
    "In what ways does Daisy Buchanan embody both illusion and reality?",
    "Why does Gatsby throw lavish parties, and what do they reveal about his character?"
    # "How does Fitzgerald use the contrast between East Egg and West Egg to comment on social class?",
    # "What does the billboard of Dr. T. J. Eckleburg’s eyes symbolize?",
    # "How does the weather (sunshine, rain, heat) mirror the emotional tone of key scenes?",
    # "What does Tom Buchanan represent in the context of wealth and power?",
    # "How is the American Dream portrayed as both promise and illusion in the novel?",
    # "What is the significance of Gatsby’s reunion with Daisy at Nick’s house?",
    # "How does Myrtle Wilson’s fate underscore the novel’s themes of desire and destruction?"
]

for q in test_questions:
    start = time.perf_counter()
    try:
        answer = answer_rag(q)
    except Exception as e:
        answer = f"Error: {e}"
    elapsed = time.perf_counter() - start
    print("Q:", q)
    print("A:", answer)
    print(f"→ Time: {elapsed:.3f} s")
    print("-" * 80)

