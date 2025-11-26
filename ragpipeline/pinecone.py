from pinecone import Pinecone, ServerlessSpec
import os
from config import RESUME_INDEX_NAME

# ---------- Step 1–3: Initialize Pinecone, ensure index, get handle ----------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
if RESUME_INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=RESUME_INDEX_NAME,
        dimension=384,            # must match dense model
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(RESUME_INDEX_NAME)
