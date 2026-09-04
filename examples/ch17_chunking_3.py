# Chapter 17 — 17.3 Chunking
import re

CLAUSE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.*)$")


def clause_chunks(text: str, doc_id: str) -> list[dict]:
    """One chunk per numbered clause, number preserved."""
    out, number, title, body = [], None, "", []
    for line in text.splitlines():
        m = CLAUSE.match(line.strip())
        if m:
            if number:
                out.append({
                    "document_id": doc_id,
                    "clause": number,
                    "heading": title,
                    "content": f"{number} {title}\n"
                               + "\n".join(body),
                })
            number, title, body = m.group(1), m.group(2), []
        elif number:
            body.append(line)
    if number:
        out.append({
            "document_id": doc_id, "clause": number,
            "heading": title,
            "content": f"{number} {title}\n" + "\n".join(body),
        })
    return out
