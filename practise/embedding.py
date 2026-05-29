from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer('all-MiniLM-L6-v2')

print("Model loaded!")

sentences = [
    "Machine Learning is powerful",
    "Artificial Intelligence is changing the world"
]

print("Creating embeddings...")

embeddings = model.encode(sentences)

print("Embeddings created!")

print(len(embeddings))
print(len(embeddings[0]))
