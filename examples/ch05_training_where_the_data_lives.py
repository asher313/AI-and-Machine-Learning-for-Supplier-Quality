# Chapter 5 — 5.5 Training Where the Data Lives
# Training inside the database (no data extraction)
from hana_ml.dataframe import ConnectionContext
from hana_ml.algorithms.pal.trees import (
    RandomForestClassifier,
)

from sqm_ai.settings import get_settings

s = get_settings()
cc = ConnectionContext(
    address=s.hana_host,
    port=30015,
    user=s.hana_user,
    password=s.hana_password.get_secret_value(),
)

# A handle, not a pandas frame: nothing is downloaded
# label sev3_next_90d is defined in Chapter 10
df = cc.table("SUPPLIER_FEATURES", schema="SQM")

rfc = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
)

rfc.fit(
    data=df,
    key="supplier_id",
    features=["fpy", "otd", "audit_score", "ncr_count"],
    label="sev3_next_90d",
)

# Scoring also happens in the database
scored = rfc.predict(cc.table("SUPPLIER_SCORING", schema="SQM"))
print(scored.head(5).collect())   # collect() downloads
