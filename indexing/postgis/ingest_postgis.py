"""PostGIS ingestion — load GeoJSON collections into PostgreSQL with spatial indexes."""

import json
import psycopg2
from pathlib import Path


def ingest_collection(filepath, dsn):
    """Load a single GeoJSON collection into a PostGIS table.

    Creates a table named after the file, with a geometry column (SRID 4326),
    all properties as text/numeric columns, and a GIST spatial index.

    Args:
        filepath: Path to the collection JSON file
        dsn: PostgreSQL connection string
    """
    with open(filepath) as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    features = data.get("geojson", {}).get("features", [])
    table_name = Path(filepath).stem

    if not features:
        return

    # Infer columns from first feature's properties
    sample_props = features[0].get("properties", {})
    columns = list(sample_props.keys())

    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    # Create table
    col_defs = ", ".join(f'"{col}" TEXT' for col in columns)
    cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    cur.execute(f'CREATE TABLE "{table_name}" (id SERIAL PRIMARY KEY, geom GEOMETRY(Geometry, 4326), {col_defs})')

    # Insert features
    for feat in features:
        geom = json.dumps(feat.get("geometry"))
        props = feat.get("properties", {})
        values = [str(props.get(col, "")) if props.get(col) is not None else None for col in columns]

        placeholders = ", ".join(["%s"] * len(columns))
        cur.execute(
            f'INSERT INTO "{table_name}" (geom, {", ".join(f"{c}" for c in columns)}) '
            f'VALUES (ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), {placeholders})',
            [geom] + values
        )

    # Create spatial index
    cur.execute(f'CREATE INDEX ON "{table_name}" USING GIST (geom)')

    conn.commit()
    cur.close()
    conn.close()


def create_metadata_table(collections_dir, dsn):
    """Create a dataset_metadata table cataloguing all ingested collections."""
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS dataset_metadata (
            dataset_key TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            feature_count INTEGER,
            geometry_types TEXT[],
            property_names TEXT[],
            municipality TEXT
        )
    """)

    for filepath in sorted(Path(collections_dir).glob("*.json")):
        with open(filepath) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        features = data.get("geojson", {}).get("features", [])
        props = list(features[0].get("properties", {}).keys()) if features else []
        geom_types = list(set(f.get("geometry", {}).get("type", "") for f in features[:100]))

        cur.execute(
            "INSERT INTO dataset_metadata VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [filepath.stem, meta.get("title"), meta.get("description"),
             len(features), geom_types, props, None]
        )

    conn.commit()
    cur.close()
    conn.close()
