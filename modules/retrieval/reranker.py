import numpy as np

def cosine_similarity(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a,b) / ((np.linalg.norm(a))* np.linalg.norm(b))

def rerank(query_emb,docs,doc_embs,top_k=5):
    print("🎯 Running cosine similarity reranker") #checking
    scores = []

    for i, emb in enumerate(doc_embs):
        score = cosine_similarity(query_emb,emb)
        scores.append((score, docs[i]))

    #sorts scroes in descending order and returns top_k docs
    scores.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scores[:top_k]]

