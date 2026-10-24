import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

sentences = [
    "What if I hand in my homework after the deadline?",
    "Assignments submitted late lose 10% of their marks for each day.",
    "The course uses Python and Jupyter notebooks.",
]

result = client.models.embed_content(model="gemini-embedding-001", contents=sentences)
vectors = [e.values for e in result.embeddings]

print("Numbers per sentence:", len(vectors[0]))
print("First 5 numbers of sentence 1:", vectors[0][:5])


def similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    size_a = sum(x * x for x in a) ** 0.5
    size_b = sum(x * x for x in b) ** 0.5
    return dot / (size_a * size_b)


print("Question vs late policy:", round(similarity(vectors[0], vectors[1]), 3))
print("Question vs software:   ", round(similarity(vectors[0], vectors[2]), 3))