import os
import uuid
import logging
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import UploadResponse, QueryRequest, QueryResponse, RetrievedChunk
from .utils import is_allowed, extract_text_from_pdf, extract_text_from_txt, chunk_text
from .vectorstore import SimpleVectorStore

# dotenv (optional)
from dotenv import load_dotenv
load_dotenv()

# Try to import LCEL (langchain-core) and the Gemini client
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LCEL_AVAILABLE = True
except Exception:
    LCEL_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Mini RAG API (final)")

# CORS for Streamlit UI (port 8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vs = SimpleVectorStore()
MAX_FILE_SIZE = 40 * 1024 * 1024  # 40 MB

# --- Lazy reranker state (safe single-attempt loader) ---
_reranker = None
_reranker_loaded_once = False
_reranker_failed = False

def get_reranker():
    """
    Initialize cross-encoder reranker ONCE when first requested by a query.
    If initialization fails, mark _reranker_failed True and never retry.
    Returns reranker instance or None.
    """
    global _reranker, _reranker_loaded_once, _reranker_failed

    # If already loaded successfully, return it.
    if _reranker_loaded_once:
        return _reranker

    # If failed before, do not retry.
    if _reranker_failed:
        return None

    # Try to load now (this is done only once, on-demand).
    try:
        logger.info("Loading CrossEncoder reranker (on-demand)...")
        from .reranker import CrossEncoderReranker  # local import to avoid top-level heavy load
        _reranker = CrossEncoderReranker()
        _reranker_loaded_once = True
        logger.info("Reranker initialized successfully.")
        return _reranker
    except Exception as e:
        logger.exception("Reranker failed to load (will not retry): %s", e)
        _reranker_failed = True
        _reranker = None
        return None

# --- Mock LLM for offline testing ---
class MockLLM:
    def __init__(self, **kwargs):
        pass

    def invoke(self, inputs: dict):
        # Simple mock: try to find a snippet line containing query tokens
        q = inputs.get("question", "") or ""
        snippets = inputs.get("snippets", "") or ""
        q_words = [w.lower().strip() for w in q.split() if w.strip()]
        for line in snippets.splitlines():
            if not line.strip():
                continue
            for w in q_words[:3]:
                if w and w in line.lower():
                    return f"Mock answer based on snippet: {line[:400]}"
        return "I don't know based on the provided documents. (mock)"

def get_llm():
    """
    Returns Gemini LLM if configured and available; otherwise MockLLM.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key and GEMINI_AVAILABLE and LCEL_AVAILABLE:
        try:
            # pass api_key explicitly to ensure the library receives it
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, api_key=api_key)
        except Exception as e:
            logger.exception("Failed to initialize Gemini LLM; falling back to MockLLM: %s", e)
            return MockLLM()
    else:
        if not api_key:
            logger.warning("No GOOGLE_API_KEY/GEMINI_API_KEY found — using MockLLM")
        else:
            logger.warning("Gemini or langchain-core not available — using MockLLM")
        return MockLLM()

@app.get("/ready")
def ready():
    """
    Readiness endpoint.
    - ready: server up
    - reranker_loaded: whether reranker was already loaded (True only after first successful load)
    - reranker_failed: whether initial load attempt failed (True if failed and no retry will be attempted)
    Note: calling /ready does NOT trigger loading the reranker.
    """
    return {
        "ready": True,
        "reranker_loaded": _reranker_loaded_once,
        "reranker_failed": _reranker_failed
    }

@app.post('/upload', response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    try:
        filename = file.filename
        if not is_allowed(filename):
            raise HTTPException(status_code=400, detail="File type not allowed. Allowed: .pdf, .txt, .md")
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")

        docs_to_add = []
        if filename.lower().endswith('.pdf'):
            pages = extract_text_from_pdf(contents)
            for page_no, text in pages:
                if not text.strip():
                    continue
                chunks = chunk_text(text)
                for i, c in enumerate(chunks):
                    docs_to_add.append({
                        "chunk_id": str(uuid.uuid4()),
                        "text": c,
                        "metadata": {"filename": filename, "page": page_no, "chunk": i}
                    })
        else:
            pages = extract_text_from_txt(contents)
            for page_no, text in pages:
                chunks = chunk_text(text)
                for i, c in enumerate(chunks):
                    docs_to_add.append({
                        "chunk_id": str(uuid.uuid4()),
                        "text": c,
                        "metadata": {"filename": filename, "page": page_no, "chunk": i}
                    })

        if not docs_to_add:
            return UploadResponse(success=False, message="No text extracted", uploaded_file=filename)

        vs.add_documents(docs_to_add)
        return UploadResponse(success=True, message=f"Indexed {len(docs_to_add)} chunks", uploaded_file=filename)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in upload: %s", e)
        return UploadResponse(success=False, message="Internal server error (see logs)", uploaded_file=None)

@app.post('/query', response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    try:
        # 1) First-stage retrieval: get a candidate pool from FAISS (larger than top_k)
        initial_pool = max(10, req.top_k * 4)
        candidates = vs.retrieve(req.query, top_k=initial_pool)

        if not candidates:
            return QueryResponse(answer="", confidence=0.0, retrieved=[], safe=False, message="No documents indexed yet")

        # 2) Rerank: attempt to get reranker (on-demand). If it exists and loaded, use it.
        rr = get_reranker()
        if rr is not None:
            try:
                final_candidates = rr.rerank(req.query, candidates, top_k=req.top_k)
            except Exception as e:
                logger.exception("Reranker invocation failed; falling back to FAISS score sort: %s", e)
                final_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:req.top_k]
        else:
            # reranker not loaded or failed — fallback to FAISS scores
            final_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:req.top_k]

        if not final_candidates:
            return QueryResponse(answer="", confidence=0.0, retrieved=[], safe=False, message="No candidates after reranking")

        # 3) Safety: if top score (rerank_score if present) is below threshold, refuse
        top_score = final_candidates[0].get('rerank_score', final_candidates[0].get('score', 0.0))
        CONF_THRESHOLD = 0.15
        if top_score < CONF_THRESHOLD:
            return QueryResponse(answer="", confidence=float(top_score), retrieved=[], safe=False, message="Query seems unrelated to uploaded documents. Please upload relevant documents first.")

        # 4) Build snippets and prepare prompt
        snippets = "\n\n".join([f"[CHUNK {r['chunk_id']}] {r['text']}" for r in final_candidates])

        prompt_text = (
            "You are a helpful assistant. Use only the provided document snippets to answer the question.\n"
            "If the answer is not contained in the snippets, say 'I don't know based on the provided documents.'\n"
            "Cite chunks inline like (CHUNK_ID).\n\n"
            "DOCUMENTS:\n{snippets}\n\nQUESTION:\n{question}\n\nANSWER:"
        )

        # 5) Call LLM using LCEL pipeline if available, otherwise MockLLM
        llm = get_llm()
        if LCEL_AVAILABLE and GEMINI_AVAILABLE and not isinstance(llm, MockLLM):
            try:
                prompt = PromptTemplate(input_variables=["snippets", "question"], template=prompt_text)
                parser = StrOutputParser()
                chain = prompt | llm | parser
                resp = chain.invoke({"snippets": snippets, "question": req.query})
            except Exception as e:
                logger.exception("LCEL pipeline failed; falling back to direct llm.invoke: %s", e)
                resp = llm.invoke({"snippets": snippets, "question": req.query})
        else:
            resp = llm.invoke({"snippets": snippets, "question": req.query})

        # 6) Confidence: average rerank_score if present else original scores
        scores_for_conf = [c.get('rerank_score', c.get('score', 0.0)) for c in final_candidates]
        avg_score = sum(scores_for_conf) / len(scores_for_conf)

        retrieved_out = [
            RetrievedChunk(
                chunk_id=r['chunk_id'],
                text=r['text'][:400],
                filename=r['metadata'].get('filename',''),
                score=float(r.get('rerank_score', r.get('score', 0.0)))
            )
            for r in final_candidates
        ]

        return QueryResponse(answer=str(resp).strip(), confidence=float(avg_score), retrieved=retrieved_out, safe=True, message=None)
    except Exception as e:
        logger.exception("Error in query: %s", e)
        return QueryResponse(answer="", confidence=0.0, retrieved=[], safe=False, message="Internal server error (see logs)")
