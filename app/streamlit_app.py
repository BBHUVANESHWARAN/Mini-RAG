import streamlit as st
import requests
import os
import time
from requests.exceptions import RequestException, ConnectionError, ReadTimeout

API_BASE = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Mini RAG", layout="wide")
st.title("Mini RAG")

st.sidebar.header("Actions")
mode = st.sidebar.selectbox("Mode", ["Upload Document", "Ask a Question", "Health"])

def api_is_up(retries=5, backoff=1.5, timeout=6):
    """
    Robust health check to wait while backend finishes loading models.
    """
    url = f"{API_BASE}/ready"
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                # if ready endpoint returns JSON, parse and return
                try:
                    data = r.json()
                    return True, data
                except Exception:
                    return True, {"ready": True}
            else:
                time.sleep(backoff * attempt)
        except (ConnectionError, ReadTimeout, RequestException):
            time.sleep(backoff * attempt)
    return False, None

api_ok, api_info = api_is_up(retries=4, backoff=1.5, timeout=6)

if not api_ok:
    st.warning(
        f"RAG API not reachable at {API_BASE}. "
        "Make sure the backend is running. If the backend is starting it may be loading models; try again in 20-60s."
    )
    st.stop()

# Show reranker status (if available)
reranker_loaded = False
if api_info and isinstance(api_info, dict):
    reranker_loaded = api_info.get("reranker_loaded", False)

if mode == "Upload Document":
    st.write("Upload a PDF / TXT / MD file to index into the RAG system.")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf","txt","md"])
    if uploaded_file is not None:
        if st.button("Upload to RAG API"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                with st.spinner("Uploading..."):
                    resp = requests.post(f"{API_BASE}/upload", files=files, timeout=120)
                if resp.status_code == 200:
                    st.success("Uploaded successfully")
                    try:
                        st.json(resp.json())
                    except Exception:
                        st.write(resp.text)
                else:
                    st.error(f"Upload failed: {resp.status_code}")
                    st.text(resp.text)
            except ReadTimeout:
                st.error("Upload timed out — backend is busy (models may be loading). Try again in a minute.")
            except ConnectionError:
                st.error(f"Could not connect to API at {API_BASE}. Is the backend running?")
            except Exception as e:
                st.error("Error uploading file")
                st.exception(e)

elif mode == "Ask a Question":
    if not reranker_loaded:
        st.info("Note: reranker not yet loaded. The system will use FAISS scores until the reranker finishes initializing.")
    st.write("Ask a question that should be answered using the uploaded documents.")
    query = st.text_area("Question", height=120)
    top_k = st.number_input("Top K", min_value=1, max_value=10, value=5)
    if st.button("Get Answer"):
        payload = {"query": query, "top_k": int(top_k)}
        try:
            with st.spinner("Querying RAG API..."):
                resp = requests.post(f"{API_BASE}/query", json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("safe", False):
                    st.warning(data.get("message", "Query was rejected or unrelated to documents."))
                st.subheader("Answer")
                st.write(data.get("answer", ""))
                st.info(f"Confidence: {data.get('confidence', 0):.3f}")

                st.subheader("Final Output")
                for r in data.get("retrieved", []):
                    with st.expander(f"Chunk {r.get('chunk_id')} — score: {r.get('score')}"):
                        st.write(r.get('text'))
                        st.caption(f"File: {r.get('filename')}")
            else:
                st.error(f"Query failed: {resp.status_code}")
                st.text(resp.text)
        except ReadTimeout:
            st.error("Query timed out — backend may be busy. Try again in a moment.")
        except ConnectionError:
            st.error(f"Could not connect to API at {API_BASE}. Is the backend running?")
        except Exception as e:
            st.error("Error querying RAG API")
            st.exception(e)

else:  # Health check
    st.write("Health / Info")
    try:
        r = requests.get(f"{API_BASE}/openapi.json", timeout=10)
        if r.status_code == 200:
            st.success("API reachable")
            try:
                st.json(r.json())
            except Exception:
                st.text(r.text)
        else:
            st.error("API not reachable")
            st.text(r.text)
    except Exception as e:
        st.error("API not reachable")
        st.exception(e)
