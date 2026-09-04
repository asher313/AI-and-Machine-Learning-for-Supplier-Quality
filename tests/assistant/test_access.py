# tests/assistant/test_access.py
from sqm_ai.assistant.access import AssistantUser
from sqm_ai.assistant.retrieve import fetch_chunks


def test_restricted_chunk_is_never_retrieved(corpus):
    """Ask the question a NL-MERLIN chunk answers, as a
    user cleared only for NL-KESTREL."""
    user = AssistantUser("u-114", ("NL-KESTREL",))
    chunks = fetch_chunks(corpus.merlin_question, user)
    ids = {c["id"] for c in chunks}
    assert corpus.merlin_chunk_id not in ids
    assert chunks, "filter must not empty the result"
