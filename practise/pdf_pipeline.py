from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai

import os

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
)
# Read PDF
reader = PdfReader("Swarna new.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Total Chunks:", len(chunks))

print("\nFirst Chunk:\n")
print(chunks[0])

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("Embedding Shape:")
print(len(embeddings))
print(len(embeddings[0]))

import faiss
import numpy as np

embeddings = np.array(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS Index Created")

query = input("Ask a question: ")

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding)

distances, indices = index.search(query_embedding, 5)

print("\nRelevant Chunks:\n")

retrieved_text = ""

for idx in indices[0]:
    retrieved_text += chunks[idx] + "\n\n"

prompt = f"""
Use the following context to answer the question.

Context:
{retrieved_text}

Question:
{query}

Answer:
"""
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print("\nAnswer:\n")
print(response.text)