from config import RERANK_MODEL_NAME
from sentence_transformers import CrossEncoder
rerank_model = CrossEncoder(RERANK_MODEL_NAME)
def rerank_docs_crossencoder(results, query):
    if not results:
        return []
    pairs = [(query, r['preview']) for r in results]
    scores = rerank_model.predict(pairs)
    rescored = []
    for r, s in zip(results, scores):
        r2 = dict(r)
        r2['rerank_score'] = float(s)
        rescored.append(r2)
    return sorted(rescored, key = lambda x: x['rerank_score'], reverse = True)
    # return scores