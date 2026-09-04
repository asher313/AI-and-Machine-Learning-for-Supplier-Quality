# Chapter 17 — 17.8 Query Expansion and HyDE
# src/sqm_ai/retrieval/expand.py
from sqm_ai.llm import MODELS, client

EXPAND = (
    "Rewrite the engineer's question three different ways "
    "to improve document search over an aerospace quality "
    "corpus. Vary the vocabulary; keep every part number, "
    "clause number, and supplier id exactly as written. "
    "Output only the three rewrites, one per line.")


def expand(question: str) -> list[str]:
    """The original question plus three rephrasings."""
    response = client.messages.create(
        model=MODELS["fast"],
        max_tokens=300,
        system=EXPAND,
        messages=[{"role": "user", "content": question}],
    )
    text = next(
        b.text for b in response.content if b.type == "text")
    lines = [ln.strip() for ln in text.splitlines()
             if ln.strip()]
    return [question] + lines[:3]


# Chapter 17 — 17.8 Query Expansion and HyDE (continued)
# src/sqm_ai/retrieval/expand.py, continued.
HYDE = (
    "Write one short paragraph that plausibly answers the "
    "question, in the style of an aerospace quality manual. "
    "It will be used only as a search probe.")


def hyde_vector(question: str, embedder):
    """Embed a fabricated answer, not the question."""
    response = client.messages.create(
        model=MODELS["fast"], max_tokens=250,
        system=HYDE,
        messages=[{"role": "user", "content": question}],
    )
    draft = next(
        b.text for b in response.content if b.type == "text")
    return embedder.encode(
        [draft], normalize_embeddings=True)[0]
