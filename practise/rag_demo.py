from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Sample chunks
chunks = [
    "Machine Learning is a branch of AI",
    "Football is a popular sport",
    "Deep Learning uses neural networks"
]

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = model.encode(chunks)

embeddings = np.array(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# User question
query = input("Ask a question: ")

# Convert question to embedding
query_embedding = model.encode([query])

query_embedding = np.array(query_embedding)

# Search
k = 1

distances, indices = index.search(query_embedding, k)

retrieved_chunk = chunks[indices[0][0]]

# Simulated answer generation
print("\nRetrieved Information:")
print(retrieved_chunk)

print("\nGenerated Answer:")
print(f"Based on the document: {retrieved_chunk}")