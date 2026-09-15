"""Import every sqm_ai module.

The listings in this repository are extracted verbatim from
the book, so a module is only importable when the third-party
packages that chapter installed are present. Each module below
therefore names the packages it needs, and the test is skipped
rather than failed when one of them is missing.
"""

import importlib

import pytest

# module -> third-party packages that must be importable first
MODULES: dict[str, tuple[str, ...]] = {
    "sqm_ai": (),
    "sqm_ai.agent": (),
    "sqm_ai.agent.loop": ("anthropic", "structlog"),
    "sqm_ai.agent.replay": ("anthropic", "pydantic"),
    "sqm_ai.agent.tools": ("anthropic", "structlog"),
    "sqm_ai.anomalies": (
        "pandas",
        "numpy",
        "scipy",
        "sqlalchemy",
        "structlog",
    ),
    "sqm_ai.assistant": (),
    "sqm_ai.assistant.access": (),
    "sqm_ai.assistant.answer": ("anthropic", "pydantic"),
    "sqm_ai.assistant.evaluation": (
        "ragas",
        "langchain_community",
    ),
    "sqm_ai.assistant.judge": ("anthropic", "pydantic"),
    "sqm_ai.assistant.metrics": ("numpy",),
    "sqm_ai.assistant.prompts": (),
    "sqm_ai.assistant.retrieve": (
        "psycopg",
        "sentence_transformers",
    ),
    "sqm_ai.assistant.run_build4": (
        "anthropic",
        "pydantic",
        "psycopg",
    ),
    "sqm_ai.assistant.schema": ("pydantic",),
    "sqm_ai.assistant.validate": ("anthropic", "pydantic"),
    "sqm_ai.assistant.versioning": ("psycopg",),
    "sqm_ai.aws": (),
    "sqm_ai.aws.enclave": ("anthropic", "boto3"),
    "sqm_ai.aws.metrics": ("boto3",),
    "sqm_ai.aws.secrets": ("boto3",),
    "sqm_ai.aws.storage": ("boto3",),
    "sqm_ai.bakeoff": ("pandas", "numpy", "sklearn"),
    "sqm_ai.build1": (),
    "sqm_ai.build1.evaluation": ("pandas", "numpy"),
    "sqm_ai.build1.explain": ("pandas", "numpy", "shap"),
    "sqm_ai.build1.features": ("pandas", "sqlalchemy"),
    "sqm_ai.build1.labels": ("numpy", "pandas"),
    "sqm_ai.build1.score_suppliers": (
        "pandas",
        "joblib",
        "sqlalchemy",
    ),
    "sqm_ai.build1.train": (
        "pandas",
        "numpy",
        "xgboost",
        "sklearn",
        "scipy",
        "joblib",
    ),
    "sqm_ai.car": (),
    "sqm_ai.car.graph": ("langgraph.checkpoint.sqlite",),
    "sqm_ai.car.measure": ("rapidfuzz",),
    "sqm_ai.car.memory": ("psycopg", "numpy"),
    "sqm_ai.car.nodes": ("langgraph",),
    "sqm_ai.car.redact": (),
    "sqm_ai.car.root_cause": ("anthropic", "pydantic"),
    "sqm_ai.car.run_build5": (
        "langgraph.checkpoint.sqlite",
        "anthropic",
        "pydantic",
    ),
    "sqm_ai.car.state": (),
    "sqm_ai.car.validate": ("pydantic", "langgraph"),
    "sqm_ai.ci": (),
    "sqm_ai.ci.gate": (),
    "sqm_ai.ci.pipeline": (),
    "sqm_ai.cnc": (),
    "sqm_ai.cnc.gate": (),
    "sqm_ai.cnc.edge": ("numpy", "onnxruntime"),
    "sqm_ai.cnc.export": (
        "torch",
        "numpy",
        "onnx",
        "onnxruntime",
    ),
    "sqm_ai.cnc.features": ("numpy", "scipy"),
    "sqm_ai.cnc.preprocessing": ("numpy", "pandas", "sklearn"),
    "sqm_ai.cnc.run_build2": (
        "torch",
        "onnx",
        "onnxruntime",
        "xgboost",
    ),
    "sqm_ai.cnc.synthetic": ("numpy", "pandas"),
    "sqm_ai.cnc.train_stage1": (
        "numpy",
        "pandas",
        "xgboost",
        "sklearn",
    ),
    "sqm_ai.cnc.train_stage2": ("torch", "numpy"),
    "sqm_ai.conversation": ("anthropic", "pydantic_settings"),
    "sqm_ai.dl": (),
    "sqm_ai.dl.architectures": ("torch",),
    "sqm_ai.dl.cnc_cnn": ("torch",),
    "sqm_ai.dl.data": ("torch", "numpy"),
    "sqm_ai.dl.models": ("torch",),
    "sqm_ai.dl.ncr_finetune": (
        "torch",
        "transformers",
        "datasets",
        "sklearn",
    ),
    "sqm_ai.dl.ncr_zero_shot": ("torch", "transformers"),
    "sqm_ai.dl.risk_mlp": ("torch", "numpy", "pandas"),
    "sqm_ai.dl.train": ("torch",),
    "sqm_ai.errors": ("structlog",),
    "sqm_ai.extract": (
        "pandas",
        "yaml",
        "sqlalchemy",
        "structlog",
    ),
    "sqm_ai.features": (
        "pandas",
        "numpy",
        "sklearn",
        "sqlalchemy",
        "structlog",
    ),
    "sqm_ai.gateway": ("anthropic", "structlog"),
    "sqm_ai.gateway.audit": ("psycopg",),
    "sqm_ai.gateway.detectors": (),
    "sqm_ai.gateway.guardrails": ("pydantic",),
    "sqm_ai.gateway.policy": (),
    "sqm_ai.gateway.router": ("anthropic",),
    "sqm_ai.hana": ("pandas", "hdbcli", "structlog"),
    "sqm_ai.llm": ("anthropic", "structlog"),
    "sqm_ai.log": ("structlog",),
    "sqm_ai.ncr": ("pydantic",),
    "sqm_ai.retrieval": (),
    "sqm_ai.retrieval.chunking": (),
    "sqm_ai.retrieval.embed": ("sentence_transformers", "numpy"),
    "sqm_ai.retrieval.expand": ("anthropic",),
    "sqm_ai.retrieval.fuse": ("numpy", "rank_bm25"),
    "sqm_ai.retrieval.parents": ("psycopg",),
    "sqm_ai.retrieval.rerank": ("sentence_transformers",),
    "sqm_ai.retrieval.store": ("psycopg", "numpy", "pgvector"),
    "sqm_ai.retrieval.synthetic": ("numpy",),
    "sqm_ai.semantic_cache": ("numpy",),
    "sqm_ai.settings": ("pydantic", "pydantic_settings"),
    "sqm_ai.stats": ("numpy",),
    "sqm_ai.structured": (
        "anthropic",
        "pydantic",
        "pydantic_settings",
    ),
    "sqm_ai.synthetic": ("numpy", "pandas", "scipy"),
    "sqm_ai.timing": ("structlog",),
    "sqm_ai.triage": (),
    "sqm_ai.triage.classify": ("anthropic", "pydantic"),
    "sqm_ai.triage.guardrails": ("pydantic",),
    "sqm_ai.triage.pipeline": ("anthropic", "pydantic"),
    "sqm_ai.triage.review": (),
    "sqm_ai.triage.rules": (),
    "sqm_ai.triage.run_build3": ("anthropic", "pydantic"),
    "sqm_ai.triage.schema": ("pydantic",),
    "sqm_ai.triage.storage": ("sqlalchemy",),
    "sqm_ai.triage.trace": ("structlog",),
    "sqm_ai.triage_intro": (),
    "sqm_ai.verify_synthetic": ("numpy", "pandas", "scipy"),
}


@pytest.mark.parametrize("name,deps", sorted(MODULES.items()))
def test_module_imports(name: str, deps: tuple[str, ...]):
    for dep in deps:
        pytest.importorskip(dep)
    importlib.import_module(name)


def test_every_module_is_listed():
    """No sqm_ai module escapes this file."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    pkg = root / "src" / "sqm_ai"
    found = set()
    for p in pkg.rglob("*.py"):
        rel = p.relative_to(pkg.parent).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        found.add(".".join(parts))
    assert found - set(MODULES) == set()
