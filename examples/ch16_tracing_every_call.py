# Chapter 16 — 16.7 Tracing Every Call
result, trace = traced(
    lambda: call_model(MODELS["fast"], ncr, nbrs),
    ncr.defect_description,
    ncr_id=ncr.ncr_id,
    stage="fast",
)
