"""Qdrant ingestion script for geospatial features.

Loads dense embeddings, sparse vectors, and sentence text,
then creates a Qdrant collection with dual-vector search
and geo-indexed payloads.
"""

import json
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from qdrant_client import QdrantClient, models


COLLECTION_NAME = "dk_geospatial"
EMBEDDING_DIM = 1024
BATCH_SIZE = 500


def create_collection(client):
    """Create Qdrant collection with dense + sparse vectors and geo index."""
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="location",
        field_schema=models.PayloadSchemaType.GEO,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="collection",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def ingest(sentences_dir, embeddings_dir, qdrant_path):
    """Main ingestion pipeline.

    Args:
        sentences_dir: Path to directory with per-collection sentence JSON files
        embeddings_dir: Path to directory with dense parquet + sparse JSON files
        qdrant_path: Path for Qdrant local storage
    """
    # Load sentence lookup (global_id → text + metadata)
    lookup = {}
    for filepath in sorted(Path(sentences_dir).glob("*.json")):
        with open(filepath) as f:
            for entry in json.load(f):
                gid = entry["properties"].get("global_id")
                if gid is not None:
                    lookup[gid] = {
                        "text": entry["text"],
                        "collection": entry.get("collection", filepath.stem),
                        "properties": entry["properties"],
                    }

    # Load dense embeddings from parquet shards
    all_ids, all_vecs, all_lats, all_lons = [], [], [], []
    for path in sorted(Path(embeddings_dir).glob("dense_*.parquet")):
        t = pq.read_table(path)
        all_ids.extend(t.column("global_id").to_pylist())
        all_lats.extend(t.column("lat").to_pylist())
        all_lons.extend(t.column("lon").to_pylist())
        vecs = np.column_stack([t.column(f"emb_{i}").to_numpy() for i in range(EMBEDDING_DIM)])
        all_vecs.append(vecs)
    embeddings = np.concatenate(all_vecs)

    # Load sparse vectors
    sparse = {}
    for path in sorted(Path(embeddings_dir).glob("bm25_sparse_*.json")):
        with open(path) as f:
            for entry in json.load(f):
                sparse[entry["global_id"]] = {
                    "indices": entry["indices"],
                    "values": entry["values"],
                }

    # Create collection and ingest
    client = QdrantClient(path=str(qdrant_path))
    create_collection(client)

    points = []
    for idx, gid in enumerate(all_ids):
        if gid not in lookup:
            continue

        entry = lookup[gid]
        payload = {
            "text": entry["text"],
            "global_id": gid,
            "collection": entry["collection"],
        }
        if all_lats[idx] is not None and all_lons[idx] is not None:
            payload["location"] = {"lat": float(all_lats[idx]), "lon": float(all_lons[idx])}

        vectors = {"dense": embeddings[idx].tolist()}
        if gid in sparse:
            vectors["sparse"] = models.SparseVector(
                indices=sparse[gid]["indices"],
                values=sparse[gid]["values"],
            )

        points.append(models.PointStruct(id=gid, vector=vectors, payload=payload))

        if len(points) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    client.close()
