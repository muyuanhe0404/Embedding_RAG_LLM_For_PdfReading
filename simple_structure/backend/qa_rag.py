# qa_rag.py

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"     

import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Retrieve
# load chunk and index
index = faiss.read_index("data/gatsby_index.faiss")
with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# load embedding model 
embedding_model = SentenceTransformer("all-mpnet-base-v2")

def retrieve(question: str, top_k: int = 5):
    # get the most relative top chunk
    # vectorize
    q_vec = embedding_model.encode(question, convert_to_numpy=True)
    # check the nearest
    distances, indices = index.search(q_vec.reshape(1, -1), top_k)
    # take nearest chunk
    return [chunks[i] for i in indices[0]]



# generation
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    framework="pt",       # PyTorch
    device_map="auto"    
)

def answer_rag(question: str, top_k: int = 5, max_length: int = 256):
    # Retrieve
    docs = retrieve(question, top_k=top_k)
    # join
    context = "\n\n".join([d.replace("\n"," ") for d in docs])

    # prompt
    prompt = (
        "Here is some context from The Great Gatsby:\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer in a complete sentence, in your own words:"
    )
    # generator
    output = generator(
        prompt,
        max_length=max_length,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    return output[0]["generated_text"]
