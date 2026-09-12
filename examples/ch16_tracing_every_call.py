from sqm_ai.llm import MODELS
from sqm_ai.triage.classify import call_model

# Supply an NCRInput and confirmed prior examples as described in Chapter 16.
traces = []
result, response = call_model(MODELS["fast"], ncr, nbrs, traces=traces)
trace_ids = [trace.trace_id for trace in traces]
