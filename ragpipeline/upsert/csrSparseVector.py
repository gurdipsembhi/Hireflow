from typing import List , Dict
from langchain_core.documents import Document

def csr_row_to_pinecone_sparse(csr_row) -> Dict[str, List[float]]:
    coo = csr_row.tocoo()
    return {
        "indices": coo.col.tolist(),
        "values": coo.data.astype(float).tolist()
    }