# scripts/verify_snapshot.py — run twice a year
from sqm_ai.llm import MODELS, client

live = {m.id for m in client.models.list()}
for tier, model_id in MODELS.items():
    mark = "ok " if model_id in live else "GONE"
    print(f"{mark} {tier:9s} {model_id}")

# A "GONE" line is not an emergency. It is a merge
# request that changes one constant, then re-runs the
# golden sets before anyone believes the new number.
