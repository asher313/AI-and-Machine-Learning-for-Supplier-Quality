import importlib.util
from pathlib import Path
import anthropic
import httpx2


def module():
    path = (
        Path(__file__).parents[1] / "scripts/verify_snapshot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "snapshot", path
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_metadata_pagination_and_absence_does_not_claim_retirement():
    snapshot = module()
    calls = []

    def respond(request):
        calls.append(request)
        idx = len(calls)
        row = {
            "id": f"model-{idx}",
            "type": "model",
            "created_at": "2026-01-01T00:00:00Z",
            "display_name": f"Model {idx}",
        }
        return httpx2.Response(
            200,
            json={
                "data": [row],
                "has_more": idx == 1,
                "first_id": row["id"],
                "last_id": row["id"],
            },
        )

    client = anthropic.Anthropic(
        api_key="synthetic",
        max_retries=0,
        http_client=httpx2.Client(
            transport=httpx2.MockTransport(respond)
        ),
    )
    ids = snapshot.read_live(client)
    assert (
        ids == ["model-1", "model-2"]
        and len(calls) == 2
        and calls[1].url.params["after_id"] == "model-1"
    )
    rows = snapshot.inspect_catalog(
        ids, {"known": "model-2", "unseen": "model-3"}
    )
    assert [r["status"] for r in rows] == ["LISTED", "NOT_LISTED"]
