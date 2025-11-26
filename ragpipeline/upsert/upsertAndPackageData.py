from typing import List, Dict, Any
from ragpipeline.pinecone import index
from langchain_core.documents import Document
from ragpipeline.upsert.csrSparseVector import csr_row_to_pinecone_sparse
from config import RESUME_INDEX_NAME

def step6_package_and_upsert(payload: Dict[str, Any]):
    docs: List[Document] = payload["docs"]
    dense_vectors = payload.get("dense_vectors", [])
    tfidf_matrix = payload.get("tfidf_matrix")

    if not docs or tfidf_matrix is None:
        return {"upserted": 0, "index": RESUME_INDEX_NAME}

    vectors = []

    for i, (doc, dense) in enumerate(zip(docs, dense_vectors)):
        sparse = csr_row_to_pinecone_sparse(tfidf_matrix[i])
        resume_id = doc.metadata.get("resume_id")
        vid = resume_id
        meta = {
            **{k: v for k, v in doc.metadata.items() if k != "text"},
            "text": doc.page_content[:1200],  # snippet for preview
        }
        vectors.append(
            {
                "id": vid,
                "values": dense,
                "sparse_values": sparse,
                "metadata": meta,
            }
        )
    if vectors:
        index.upsert(vectors=vectors)
    return {"upserted": len(vectors), index: RESUME_INDEX_NAME}