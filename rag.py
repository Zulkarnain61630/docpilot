import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
import chromadb
from pypdf import PdfReader

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

DATA_FOLDER = Path("data")
EMBEDDING_MODEL = "gemini-embedding-001"

db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

def load_chunks():
    chunks = []
    for file in DATA_FOLDER.iterdir():
        if file.suffix.lower() == ".txt":
            text = file.read_text(encoding="utf-8")
            for paragraph in text.split("\n\n"):
                paragraph = paragraph.strip()
                if paragraph:
                    chunks.append({"source": file.name, "page": 1, "text": paragraph})
        elif file.suffix.lower() == ".pdf":
            reader = PdfReader(file)
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                for piece in chunk_text(page_text):
                    chunks.append({"source": file.name, "page": page_number, "text": piece})
    return chunks


def chunk_text(text, size=100, overlap=20):
    all_words = text.split()
    chunks = []
    start = 0
    while start < len(all_words):
        piece = all_words[start:start + size]
        chunks.append(" ".join(piece))
        if start + size >= len(all_words):
            break
        start = start + size - overlap
    return chunks


STOP_WORDS = {"the", "is", "a", "an", "of", "to", "in", "and", "what", "how",
              "do", "i", "my", "for", "are", "on", "with", "if", "it", "can"}


def words(text):
    return set(re.findall(r"\w+", text.lower())) - STOP_WORDS


def keyword_search(question, top_k=2):
    question_words = words(question)
    scored = []
    for chunk in load_chunks():
        score = len(question_words & words(chunk["text"]))
        scored.append((score, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k] if score > 0]

def embed(texts):
        result = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
        return [e.values for e in result.embeddings]


def similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    size_a = sum(x * x for x in a) ** 0.5
    size_b = sum(x * x for x in b) ** 0.5
    return dot / (size_a * size_b)


def meaning_search(question, chunks, chunk_vectors, top_k=2):
    question_vector = embed([question])[0]
    scored = []
    for chunk, vector in zip(chunks, chunk_vectors):
        scored.append((similarity(question_vector, vector), chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]

def ingest():
    existing = collection.get()["ids"]
    if existing:
        collection.delete(ids=existing)

    chunks = load_chunks()
    vectors = embed([chunk["text"] for chunk in chunks])
    ids = [f"{chunk['source']}-p{chunk['page']}-{i}" for i, chunk in enumerate(chunks)]
    collection.upsert(
        ids=ids,
        documents=[chunk["text"] for chunk in chunks],
        embeddings=vectors,
        metadatas=[{"source": chunk["source"], "page": chunk["page"]} for chunk in chunks],
    )
    print(f"Stored {len(chunks)} chunks. Database now holds {collection.count()} chunks.")

def search_db(query, top_k=3):
    query_vector = embed([query])[0]
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
    )
    found = []
    for text, meta, distance in zip(results["documents"][0],
                                    results["metadatas"][0],
                                    results["distances"][0]):
        found.append({"source": meta["source"], "page": meta["page"],
                      "text": text, "score": 1 - distance})
    return found

def list_sources():
    all_metadata = collection.get(include=["metadatas"])["metadatas"]
    return sorted({meta["source"] for meta in all_metadata})

if __name__ == "__main__":
    ingest()