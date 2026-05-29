from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai
import streamlit as st

import os

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
)

@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = get_model()

st.title("📄 RAG PDF Chatbot")

question = st.text_input("Ask a question about the PDF")

if question:
    st.write("Question:", question)

# Read PDF
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file is not None:
    # Use session state to cache document chunking and index creation
    if "file_name" not in st.session_state or st.session_state.file_name != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.split_text(text)
            
            if chunks:
                embeddings = model.encode(chunks)
                embeddings = np.array(embeddings)
                
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)
                
                st.session_state.file_name = uploaded_file.name
                st.session_state.chunks = chunks
                st.session_state.index = index
            else:
                st.session_state.file_name = None
                st.session_state.chunks = []
                st.session_state.index = None

    if st.session_state.index is not None:
        chunks = st.session_state.chunks
        index = st.session_state.index

        st.success("PDF Loaded Successfully")
        st.write("Total Chunks:", len(chunks))

        if question:
            query_embedding = model.encode([question])
            query_embedding = np.array(query_embedding)

            # Limit retrieval to available chunks (up to 5)
            k = min(5, len(chunks))
            if k > 0:
                distances, indices = index.search(query_embedding, k)

                retrieved_text = ""
                for idx in indices[0]:
                    # Prevent indexing out of bounds or -1 errors
                    if 0 <= idx < len(chunks):
                        retrieved_text += chunks[idx] + "\n\n"

                prompt = f"""
                Use the following context to answer the question.

                Context:
                {retrieved_text}

                Question:
                {question}

                Answer:
                """
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                st.subheader("Answer")
                st.write(response.text)
            else:
                st.warning("No text chunks available to query.")
    else:
        st.error("Could not extract readable text from the uploaded PDF.")