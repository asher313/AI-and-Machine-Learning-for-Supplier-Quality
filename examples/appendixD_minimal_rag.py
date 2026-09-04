# Appendix D — D.3 The Minimal RAG Pipeline (Chapter 17.2)
# The minimal RAG pipeline. Memorize the four steps.
import numpy as np
from sentence_transformers import SentenceTransformer

from sqm_ai.llm import MODELS, client

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# STEP 1 — the knowledge base, one string per chunk.
documents = [
    "AS9100 Rev D 8.4.1 requires the organization to "
    "ensure externally provided processes conform...",
    "CAR-2025-0311 root cause: worn drill fixture at "
    "supplier S-0417; containment was 100% inspection...",
    # ...thousands more
]

# STEP 2 — embed the corpus once. Shape (n_docs, 384).
doc_vectors = embedder.encode(
    documents, normalize_embeddings=True)


# STEP 3 — embed the question, take the nearest chunks.
def retrieve(question: str, k: int = 5) -> list[str]:
    q = embedder.encode([question],
                        normalize_embeddings=True)[0]
    sims = doc_vectors @ q          # cosine; normalized
    top = np.argsort(sims)[::-1][:k]
    return [documents[i] for i in top]


# STEP 4 — answer from the retrieved text, with citations.
SYSTEM = (
    "Answer using ONLY the numbered context passages. "
    "Cite every claim as [N]. If the answer is not in the "
    "context, say 'That is not in the sources I have.'")


def answer(question: str) -> str:
    chunks = retrieve(question)
    context = "\n\n".join(
        f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    response = client.messages.create(
        model=MODELS["frontier"],
        max_tokens=1_024,
        system=SYSTEM,
        messages=[{"role": "user", "content":
                   f"Context:\n{context}\n\nQ: {question}"}],
    )
    return next(
        b.text for b in response.content if b.type == "text")
