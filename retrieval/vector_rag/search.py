"""Vector search using Qdrant with dense + sparse retrieval and RRF fusion."""

from qdrant_client import QdrantClient, models


def rrf_fuse(dense_results, sparse_results, k=60, dense_w=0.6, sparse_w=0.4):
    """Reciprocal Rank Fusion of dense and sparse search results."""
    scores = {}
    hits = {}

    for rank, hit in enumerate(dense_results):
        scores[hit.id] = scores.get(hit.id, 0) + dense_w * (1.0 / (k + rank + 1))
        hits[hit.id] = hit

    for rank, hit in enumerate(sparse_results):
        scores[hit.id] = scores.get(hit.id, 0) + sparse_w * (1.0 / (k + rank + 1))
        if hit.id not in hits:
            hits[hit.id] = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(hits[pid], score) for pid, score in ranked]


def build_geo_filter(lat, lon, radius_m):
    """Build Qdrant geofilter for spatial constraint."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key="location",
                geo_radius=models.GeoRadius(
                    center=models.GeoPoint(lat=lat, lon=lon),
                    radius=radius_m,
                ),
            )
        ]
    )


def search(query_dense_vec, query_sparse_vec, client, collection_name,
           top_k=100, geo_center=None, geo_radius_m=None):
    """Hybrid dense + sparse search with optional geofiltering.

    Args:
        query_dense_vec: 1024-dim dense embedding of the query
        query_sparse_vec: SparseVector (indices + values) for the query
        client: QdrantClient instance
        collection_name: name of the Qdrant collection
        top_k: number of results to retrieve per search
        geo_center: (lat, lon) tuple for spatial filtering, or None
        geo_radius_m: radius in metres for spatial filtering, or None

    Returns:
        List of (hit, score) tuples, fused and ranked
    """
    query_filter = None
    if geo_center and geo_radius_m:
        query_filter = build_geo_filter(geo_center[0], geo_center[1], geo_radius_m)

    # Dense search (semantic similarity)
    dense_results = client.query_points(
        collection_name=collection_name,
        query=query_dense_vec,
        using="dense",
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    ).points

    # Sparse search (keyword matching)
    sparse_results = []
    if query_sparse_vec:
        sparse_results = client.query_points(
            collection_name=collection_name,
            query=query_sparse_vec,
            using="sparse",
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points

    # Fuse results
    if not sparse_results:
        return [(hit, hit.score) for hit in dense_results[:top_k]]

    return rrf_fuse(dense_results, sparse_results)[:top_k]
