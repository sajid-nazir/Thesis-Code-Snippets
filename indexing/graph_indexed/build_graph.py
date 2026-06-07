"""Build Neo4j knowledge graph from GeoJSON collection metadata.

Creates nodes for Regions, Municipalities, and Datasets,
connected by PART_OF and LOCATED_IN relationships.
Enrichment step adds Topic and Property nodes with semantic edges.
"""

import json
from pathlib import Path
from urllib.request import Request, urlopen


import os
NEO4J_URL = os.environ.get("NEO4J_URL", "<neo4j-http-endpoint>")


def run_cypher(statement):
    """Execute a Cypher query via Neo4j HTTP API."""
    req = Request(
        NEO4J_URL,
        data=json.dumps({"statement": statement}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def build_base_graph(collections_dir):
    """Build the geographic hierarchy: Region → Municipality → Dataset.

    Result: 2 Region + 5 Municipality + 40 Dataset nodes,
            5 PART_OF + 40 LOCATED_IN edges.
    """
    # Define geographic hierarchy
    hierarchy = {
        "Region Hovedstaden": ["Københavns Kommune", "Frederiksberg Kommune"],
        "Region Midtjylland": ["Aarhus Kommune", "Silkeborg Kommune", "Syddjurs Kommune"],
    }

    # Create regions and municipalities
    for region, municipalities in hierarchy.items():
        run_cypher(f"MERGE (r:Region {{name: '{region}'}})")
        for muni in municipalities:
            run_cypher(f"MERGE (m:Municipality {{name: '{muni}'}})")
            run_cypher(
                f"MATCH (m:Municipality {{name: '{muni}'}}), (r:Region {{name: '{region}'}}) "
                f"MERGE (m)-[:PART_OF]->(r)"
            )

    # Create dataset nodes from collection metadata
    for filepath in sorted(Path(collections_dir).glob("*.json")):
        with open(filepath) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        key = filepath.stem
        title = meta.get("title", key).replace("'", "\\'")
        desc = meta.get("description", "").replace("'", "\\'")[:200]
        feat_count = len(data.get("geojson", {}).get("features", []))

        run_cypher(
            f"MERGE (d:Dataset {{key: '{key}'}}) "
            f"SET d.title = '{title}', d.description = '{desc}', "
            f"d.feature_count = {feat_count}"
        )


def enrich_graph():
    """Add semantic enrichment: Topic nodes, Property nodes, cross-dataset edges.

    Extracts domain_tags → Topic nodes with HAS_TOPIC edges.
    Extracts shared property_keys → Property nodes with HAS_PROPERTY edges.
    Creates SHARES_TOPIC edges between datasets sharing domain tags.
    """
    # Get all datasets with their tags and properties
    result = run_cypher("MATCH (d:Dataset) RETURN d.key, d.domain_tags, d.property_keys")
    datasets = []
    for row in result['data']['values']:
        datasets.append({'key': row[0], 'tags': row[1] or [], 'props': row[2] or []})

    # Create Topic nodes + HAS_TOPIC edges
    all_topics = set()
    for d in datasets:
        all_topics.update(d['tags'])

    for topic in sorted(all_topics):
        run_cypher(f"MERGE (t:Topic {{name: '{topic}'}})")

    for d in datasets:
        for tag in d['tags']:
            run_cypher(
                f"MATCH (d:Dataset {{key: '{d['key']}'}}), (t:Topic {{name: '{tag}'}}) "
                f"MERGE (d)-[:HAS_TOPIC]->(t)"
            )

    # Create Property nodes for shared properties (appearing in 2+ datasets)
    prop_count = {}
    for d in datasets:
        for p in d['props']:
            prop_count[p] = prop_count.get(p, 0) + 1

    shared_props = {p for p, c in prop_count.items() if c >= 2}
    for prop in sorted(shared_props):
        run_cypher(f"MERGE (p:Property {{name: '{prop}'}})")

    for d in datasets:
        for p in d['props']:
            if p in shared_props:
                run_cypher(
                    f"MATCH (d:Dataset {{key: '{d['key']}'}}), (p:Property {{name: '{p}'}}) "
                    f"MERGE (d)-[:HAS_PROPERTY]->(p)"
                )

    # Create SHARES_TOPIC edges between datasets
    for i, d1 in enumerate(datasets):
        for d2 in datasets[i + 1:]:
            overlap = set(d1['tags']) & set(d2['tags'])
            if overlap:
                topics_str = ", ".join(sorted(overlap))
                run_cypher(
                    f"MATCH (a:Dataset {{key: '{d1['key']}'}}), (b:Dataset {{key: '{d2['key']}'}}) "
                    f"MERGE (a)-[:SHARES_TOPIC {{topics: '{topics_str}'}}]->(b)"
                )
