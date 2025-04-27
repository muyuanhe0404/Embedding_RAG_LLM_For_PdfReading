# test_retrieve.py
from qa_rag import retrieve

docs = retrieve("Who is Gatsby?", top_k=3)
for i, d in enumerate(docs, 1):
    print(f"--- chunk {i} ---")
    print(d[:400], "...\n")
