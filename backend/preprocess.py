import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import pickle
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import faiss

def load_pdf(path):
    #Extract raw text from a PDF file.
    return extract_text(path)

def chunk_text(text, chunk_size=500, overlap=100):
    #Split text into overlapping chunks
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        chunk = tokens[start:start+chunk_size]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap
    return chunks

def build_index(chunks, model_name='all-mpnet-base-v2'):
    #Compute embeddings for each chunk and build a FAISS index.
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def main():
    # check path
    os.makedirs('data', exist_ok=True)
    
    pdf_path = 'data/the-great-gatsby.pdf'
    text = load_pdf(pdf_path)
    print(f"Extracted {len(text)} characters")
    # split chunk
    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks (~{len(chunks[0].split())} words each)")

    # make and save faiss index
    index = build_index(chunks)
    faiss.write_index(index, 'data/gatsby_index.faiss')

    # save chunk
    with open('data/chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)

    print("Preprocessing complete: both saved in data/")


if __name__ == '__main__':
    main()