# Cell 2: Hybrid retrieval, links in context, prompt loading

import yaml
from config import RETRIEVAL_ALPHA, TOP_K, TFIDF_PATH, DENSE_MODEL_NAME
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from ragpipeline.pinecone import index
# ---------- Hybrid retrieval ----------
dense_model = SentenceTransformer(DENSE_MODEL_NAME)
def hybrid_query(query_text: str, alpha: float = RETRIEVAL_ALPHA, top_k: int = TOP_K):
    """
    Hybrid search combining dense (1 - alpha) and sparse (alpha) scores.

    alpha = 0.0 => dense-only
    alpha = 1.0 => sparse-only (keyword)
    """
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))

    # Load TF-IDF vectorizer trained during ingest
    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    # Dense query embedding
    q_dense = dense_model.encode([query_text], normalize_embeddings=True)[0]
    q_dense = (np.asarray(q_dense, dtype=float) * (1.0 - alpha)).tolist()

    # Sparse query vector
    q_sparse_csr = vectorizer.transform([query_text]).tocoo()
    if q_sparse_csr.nnz == 0:
        q_sparse = {"indices": [0], "values": [0.0]}
    else:
        q_sparse = {
            "indices": q_sparse_csr.col.tolist(),
            "values": (q_sparse_csr.data.astype(float) * alpha).tolist(),
        }

    # Query Pinecone
    res = index.query(
        vector=q_dense,
        sparse_vector=q_sparse,
        top_k=top_k,
        include_metadata=True,
    )

    out = []
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        text = md.get("text", "") or ""
        preview = " ".join(text.split()[:120])  # short snippet

        out.append(
            {
                "id": m["id"],                 # resume_id
                "score": float(m["score"]),    # hybrid score
                "preview": preview,
                "metadata": md,
            }
        )
    return out


