RETRIEVAL_ALPHA = 0.0
TOP_K = 8
# Dense model 
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# rerank model 
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# ---- Pinecone basics ----
RESUME_INDEX_NAME = "resume-hybrid-index"

PROMPT_PATH = 'skdhlskd'
EMBED_MODEL=DENSE_MODEL_NAME
TFIDF_PATH="artifacts/tf_idf_vectorizer_resumes.pkl"

RESUME_DIR = "resume_dir"  # your existing resume folder
ARTIFACTS_DIR = "artifacts"
ARTIFACTS_DIR_File="tf_idf_vectorizer_resumes.pkl"