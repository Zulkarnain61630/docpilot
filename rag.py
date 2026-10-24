import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

DATA_FOLDER = Path("data")
EMBEDDING_MODEL = "gemini-embedding-001"


def load_chunks():
    chunks = []
    for file in DATA_FOLDER.glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph:
                chunks.append({"source": file.name, "text": paragraph})
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

if __name__ == "__main__":
    chunks = load_chunks()
    chunk_vectors = embed([chunk["text"] for chunk in chunks])

    questions = [
        "What if I hand in my homework after the deadline?",
        "Can I use ChatGPT for my assignments?",
        "Who teaches this course?",
    ]
    for question in questions:
        print("\nQuestion:", question)
        for score, chunk in meaning_search(question, chunks, chunk_vectors):
            print(f"  {score:.3f} -> {chunk['text'][:60]}")