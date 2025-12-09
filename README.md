repo

 RAG-based API Service

A minimal Retrieval-Augmented Generation system with FastAPI, FAISS, LangChain LCEL pipeline, optional reranking, and Streamlit UI.

This project implements a compact, fully working RAG system that allows users to upload documents, index them, and ask questions answered using retrieval + an LLM (Gemini or fallback MockLLM).
It demonstrates clean architecture and includes retrieval, reranking, metadata storage, LCEL prompt pipeline, citations, and a simple confidence score.

 Features:
 1. Document Upload (/upload)

Accepts PDF, TXT, Markdown files.

Extracts text, splits into overlapping chunks.

Generates embeddings using SentenceTransformers.

Stores vectors in FAISS (local) for persistence.

Saves metadata: filename, page, chunk number, chunk_id.

 2. Query Endpoint (/query)

Retrieves top candidate chunks from FAISS.

Optionally applies second-stage reranking (CrossEncoder).

Builds a prompt with citations.

Uses LangChain LCEL pipeline (PromptTemplate → Gemini LLM → OutputParser) when available.

Returns:

LLM answer

Retrieved chunks

Confidence score

Safety indicator (unrelated query → rejected)

 3. Framework Requirement — Satisfied

This project uses LangChain LCEL pipeline:

prompt | llm | StrOutputParser()


Used automatically when langchain-core + Gemini key are available.

 4. Extra Feature

Cross-Encoder reranking (improves retrieval quality)
OR fallback to FAISS score-based sorting.

Inline citations: (CHUNK_ID) in answers.

 5. Safety

Disallows unrelated queries (similarity threshold).

Validates file types and size.

Sanitizes backend errors (no raw tracebacks returned).

 Project Structure
rag-api/
├── README.md
├── requirements.txt
├── streamlit_app.py        # User-facing frontend
├── app/
│   ├── main.py             # API logic (upload, query, LCEL pipeline)
│   ├── reranker.py         # CrossEncoder reranker (optional)
│   ├── vectorstore.py      # FAISS + embeddings
│   ├── utils.py            # parsing, chunking
│   ├── schemas.py          # Pydantic models
│   └── data/               # auto-created: faiss.index + meta.json


What’s included: 

app/

main.py — FastAPI app (endpoints, LCEL pipeline usage, lazy cross-encoder reranker loader, MockLLM fallback, CORS)

vectorstore.py — FAISS vector store + persistence + embedding generation (SentenceTransformers)

utils.py — file parsing (PDF/TXT/MD) and chunker

schemas.py — Pydantic request/response models

reranker.py — CrossEncoder reranker (CPU-safe model recommended)

data/ — generated at runtime: faiss.index, meta.json

streamlit_app.py — Streamlit frontend (upload, ask question, health)

requirements.txt — libraries required

Installation:
1. Create virtual environment
Windows PowerShell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt


Configure Gemini API Key

If you want real LLM answers (instead of MockLLM):

Windows (.env)
GOOGLE_API_KEY "your_gemini_key_here"




No key → system automatically falls back to a safe MockLLM.

Run Backend (FastAPI)

Start the API:

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000


Open:

Swagger Docs → http://127.0.0.1:8000/docs


Run Frontend (Streamlit UI)

In another terminal:

streamlit run streamlit_app.py


It will open:

http://localhost:8501


Features:

Upload documents

Ask questions

View retrieved chunks + confidence

📡 API Overview
POST /upload

Upload file → indexes chunks & embeddings.

Request

multipart/form-data
Field: file=@document.pdf

Response
{
  "success": true,
  "message": "Indexed 12 chunks",
  "uploaded_file": "test.pdf"
}

POST /query

Ask questions using RAG.

Request
{
  "query": "What are the key points?",
  "top_k": 5
}

Response
{
  "answer": "... (with citations)",
  "confidence": 0.42,
  "safe": true,
  "retrieved": [
    {
      "chunk_id": "abc-123",
      "text": "first part of document...",
      "filename": "notes.txt",
      "score": 0.78
    }
  ]
}

GET /ready

Reports:

API running

Whether reranker loaded successfully

{
  "ready": true,
  "reranker_loaded": false,
  "reranker_failed": false
}

Reranker Notes (Enhancement)

A CrossEncoder-based reranker is provided in app/reranker.py.

It loads lazily on the first query.

If loading fails on Windows CPU (common with some models), the system:

sets a permanent fail flag

falls back to FAISS score sorting

continues normally

This ensures the RAG system never crashes due to reranker issues.