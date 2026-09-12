# Chapter 17 teaching listing. Supply the inputs described in the text.
import numpy as np

def recall_at_k(question_vectors, corpus_vectors, relevant_ids, k=10):
    """Macro recall over nonempty relevance sets of corpus row indices."""
    if k < 1 or len(question_vectors) != len(relevant_ids) or not relevant_ids:
        raise ValueError("positive k and aligned nonempty evaluation required")
    values = []
    for query, relevant in zip(question_vectors, relevant_ids, strict=True):
        relevant = set(relevant)
        if not relevant or not relevant <= set(range(len(corpus_vectors))):
            raise ValueError("each question needs valid adjudicated relevant row ids")
        top = np.argsort(corpus_vectors @ query)[::-1][:k]
        values.append(len(relevant.intersection(top)) / len(relevant))
    return float(np.mean(values))
