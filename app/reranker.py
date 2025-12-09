from sentence_transformers import CrossEncoder

# This model works on CPU without meta-tensor errors
CROSS_ENCODER_MODEL = "cross-encoder/stsb-distilroberta-base"

class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")

    def rerank(self, query, candidates, top_k=5):
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, convert_to_numpy=True)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
