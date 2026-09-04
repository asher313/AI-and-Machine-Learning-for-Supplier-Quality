# Chapter 17 — 17.3 Chunking
# embedder: imports as in §17.2.
def semantic_chunk(
    text: str, threshold: float = 0.5
) -> list[str]:
    """Split where adjacent sentences stop being similar."""
    sentences = [s for s in text.split(". ") if s.strip()]
    if len(sentences) < 2:
        return [text]
    vecs = embedder.encode(
        sentences, normalize_embeddings=True)
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        if float(vecs[i - 1] @ vecs[i]) < threshold:
            chunks.append(". ".join(current))   # topic shift
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    chunks.append(". ".join(current))
    return chunks
