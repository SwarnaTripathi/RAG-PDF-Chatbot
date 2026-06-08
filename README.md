# 📄 RAG PDF Chatbot

[![Streamlit App]](https://rag-pdf-chatbot03.streamlit.app/)

An interactive Retrieval-Augmented Generation (RAG) PDF Chatbot built with **Streamlit**, **SentenceTransformers**, **FAISS**, and support for multiple LLM providers — **Google Gemini** and **Mistral AI**.

**Live App Link:** [rag-pdf-chatbot03.streamlit.app](https://rag-pdf-chatbot03.streamlit.app/)

Upload any PDF, ask questions about it, and get accurate, context-aware answers — powered by semantic search and your choice of LLM.

---

## 🚀 Key Features

- **Multi-LLM Support**: Switch between **Google Gemini** (`gemini-2.0-flash`) and **Mistral AI** (`mistral-small-latest`) directly from the sidebar.
- **LLM Abstraction Layer**: A clean `LLMService` class in `llm.py` encapsulates all provider-specific logic, making it easy to add new LLMs in the future.
- **Cached LLM Client**: Uses `@st.cache_resource` to instantiate the LLM client **once per provider** — no redundant API connections on every rerun.
- **Efficient Document Caching**: Processed chunks and the FAISS vector index are saved in `st.session_state`, so PDF analysis and embedding runs **only once** per upload.
- **Model Caching**: The `SentenceTransformer` embedding model is cached in memory with `@st.cache_resource`, eliminating reload delays.
- **Semantic Retrieval**: Uses `SentenceTransformers` (`all-MiniLM-L6-v2`) and `FAISS` (L2 similarity) to retrieve the top-matching passages for any query.
- **Robust Searching**: Dynamically limits FAISS search depth (`k = min(5, len(chunks))`) and filters invalid indices to prevent crashes on small documents.
- **Scanned PDF Handling**: Gracefully detects and reports PDFs with no extractable text instead of crashing.

---

## 🛠️ Architecture

1. **PDF Processing**: Reads text from uploaded PDF using `PyPDF2`.
2. **Text Splitting**: Chunks the document using LangChain's `RecursiveCharacterTextSplitter` (chunk size: 500 chars, overlap: 50 chars).
3. **Embeddings & Vector Indexing**: Encodes chunks into 384-dimensional vectors via `SentenceTransformer` and indexes them in FAISS.
4. **Context Retrieval**: Compares the user query embedding to the FAISS index to retrieve the most relevant passages.
5. **LLM Routing**: The `LLMService` class routes the prompt to the selected provider (Gemini or Mistral) and returns the generated answer.

---

## 📥 Installation & Setup

### Prerequisites
- Python 3.8+
- A **Gemini API Key** (from [Google AI Studio](https://aistudio.google.com/app/apikey)) — keys start with `AIzaSy`
- A **Mistral API Key** (from [Mistral Console](https://console.mistral.ai/api-keys))

### 1. Clone the Repository
```bash
git clone https://github.com/SwarnaTripathi/RAG-PDF-Chatbot.git
cd RAG-PDF-Chatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the project root:
```env
GEMINI_API_KEY=AIzaSyYour_Gemini_Key_Here
MISTRAL_API_KEY=Your_Mistral_Key_Here
```

Or set them as environment variables:
* **Windows (PowerShell):**
  ```powershell
  $env:GEMINI_API_KEY="AIzaSyYour_Gemini_Key_Here"
  $env:MISTRAL_API_KEY="Your_Mistral_Key_Here"
  ```
* **Linux / macOS:**
  ```bash
  export GEMINI_API_KEY="AIzaSyYour_Gemini_Key_Here"
  export MISTRAL_API_KEY="Your_Mistral_Key_Here"
  ```

---

## 🖥️ Running the Application

From inside the `project/` directory:
```bash
python -m streamlit run app.py
```
Or from the project root:
```bash
python -m streamlit run project/app.py
```
Then open **`http://localhost:8501`** in your browser.

---

## 📖 Usage Instructions

1. **Select LLM**: Use the sidebar dropdown to choose between **Gemini** or **Mistral**.
2. **Upload a PDF**: Click the file uploader and select your document.
3. **Wait for Indexing**: A `PDF Loaded Successfully` message will appear along with the total number of chunks.
4. **Ask a Question**: Type your question in the text box and press **Enter**.
5. **View Answer**: The chatbot retrieves the most relevant context and generates an answer using the selected LLM.
