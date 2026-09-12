"""Text chunking baselines; verify extracted headings/tables before indexing."""

import re

CLAUSE = re.compile(
    r"^(?:#{1,6}\s+)?(\d+(?:\.\d+)*)\s+([A-Za-z].*)$"
)


def recursive_chunks(text, doc_id, chunk_size=800, overlap=100):
    if not 0 <= overlap < chunk_size:
        raise ValueError("require 0 <= overlap < chunk_size")
    out = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for separator in ("\n\n", "\n", ". ", " "):
                cut = text.rfind(
                    separator,
                    start + max(overlap + 1, chunk_size // 2),
                    end,
                )
                if cut >= 0:
                    end = cut + len(separator)
                    break
        piece = text[start:end]
        if piece.strip():
            out.append(
                {
                    "document_id": doc_id,
                    "clause": None,
                    "heading": "",
                    "content": piece,
                }
            )
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return out


def clause_chunks(text, doc_id, chunk_size=800):
    sections = []
    number, title, body = None, "", []

    def flush():
        if body or number:
            sections.append(
                (
                    number,
                    title,
                    (f"{number} {title}\n" if number else "")
                    + "\n".join(body),
                )
            )

    for line in text.splitlines():
        match = CLAUSE.match(line.strip())
        if match:
            flush()
            number, title, body = (
                match.group(1),
                match.group(2),
                [],
            )
        else:
            body.append(line)
    flush()
    out = []
    for number, title, content in sections:
        for piece in recursive_chunks(
            content, doc_id, chunk_size
        ):
            piece.update(clause=number, heading=title)
            prefix = f"{number} {title}\n" if number else ""
            if prefix and not piece["content"].startswith(prefix):
                piece["content"] = prefix + piece["content"]
            out.append(piece)
    return out


def semantic_chunk(text, embedder, threshold=0.5, chunk_size=800):
    if not -1 <= threshold <= 1:
        raise ValueError("cosine threshold outside [-1,1]")
    sentences = [
        s
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        if s.strip()
    ]
    if not sentences:
        return []
    vectors = embedder.encode(
        sentences, normalize_embeddings=True
    )
    groups = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        if float(vectors[i - 1] @ vectors[i]) < threshold:
            groups.append(" ".join(current))
            current = []
        current.append(sentences[i])
    groups.append(" ".join(current))
    return [
        p["content"]
        for group in groups
        for p in recursive_chunks(group, "", chunk_size)
    ]
