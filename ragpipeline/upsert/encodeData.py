from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from config import EMBED_MODEL, TFIDF_PATH

def step5_encode(payload):
    docs = payload["docs"]
    corpus = [d.page_content for d in docs]
    embed = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    dense_vectors = embed.embed_documents(corpus)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {"docs": docs, "dense_vectors": dense_vectors, "tfidf_matrix":tfidf_matrix}