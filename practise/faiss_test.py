from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample text chunks
chunks = [
    "Machine Learning is a branch of AI",
    "Football is a popular sport",
    "Deep Learning uses neural networks"
]

# Convert chunks into embeddings
embeddings = model.encode(chunks)

# Convert to numpy array
embeddings = np.array(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Store embeddings
index.add(embeddings)

print("FAISS index created!")

# User question
query = "What is my name?"

# Convert question into embedding
query_embedding = model.encode([query])

query_embedding = np.array(query_embedding)

# Search similar chunks
k = 1

distances, indices = index.search(query_embedding, k)

print("\nMost Relevant Chunk:")
print(chunks[indices[0][0]])
