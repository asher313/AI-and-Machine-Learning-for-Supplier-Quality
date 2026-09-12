# Chapter 17 teaching listing. Supply the inputs described in the text.
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue,
    PointStruct, VectorParams,
)

qc = QdrantClient(url="http://qdrant.internal:6333")
qc.create_collection(
    collection_name="sqm_docs",
    vectors_config=VectorParams(
        size=1024, distance=Distance.COSINE),
)
qc.upsert("sqm_docs", points=[
    PointStruct(id=i, vector=v.tolist(),
                payload={"source_type": t, "text": c})
    for i, (v, t, c) in enumerate(zip(vecs, types, texts))
])
hits = qc.query_points(
    collection_name="sqm_docs",
    query=qvec.tolist(),
    query_filter=Filter(must=[FieldCondition(
        key="source_type",
        match=MatchValue(value="as9100"))]),
    limit=50,
).points
