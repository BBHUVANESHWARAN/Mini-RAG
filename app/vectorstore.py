import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "meta.json")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(DATA_DIR, exist_ok=True)

class SimpleVectorStore:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dim)  # cosine via normalized vectors (we'll normalize)
            self.metadata = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # normalize for IP-based cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        embs = embs / norms
        return embs.astype('float32')

    def add_documents(self, docs: List[Dict[str, Any]]):
        # docs: list of {"text":..., "metadata":{...}, "chunk_id":...}
        texts = [d["text"] for d in docs]
        embs = self._embed(texts)
        if self.index.ntotal == 0:
            # faiss.IndexFlatIP doesn't need training
            pass
        self.index.add(embs)
        # store metadata aligned with index
        for d in docs:
            self.metadata.append({"chunk_id": d["chunk_id"], "text": d["text"], "metadata": d["metadata"]})
        self._persist()

    def _persist(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def retrieve(self, query: str, top_k: int = 5):
        q_emb = self._embed([query])
        if self.index.ntotal == 0:
            return []
        scores, idxs = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({"chunk_id": meta["chunk_id"], "text": meta["text"], "metadata": meta["metadata"], "score": float(score)})
        return results

    def get_all_embeddings_for_ids(self, ids: List[int]) -> np.ndarray:
        # not used often; placeholder
        raise NotImplementedError

    def num_docs(self):
        return len(self.metadata)
