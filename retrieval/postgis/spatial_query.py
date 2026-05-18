"""PostGIS spatial query execution.

Replaces the Python-based query executor with PostgreSQL spatial SQL
using PostGIS extensions (ST_DWithin, ST_Distance).
"""

import psycopg2
import psycopg2.extras


def execute_spatial_query(dataset, operation, filters, spatial, dsn):
    """Execute a structured query using PostGIS.

    Args:
        dataset: table name in PostgreSQL
        operation: one of count, list, nearest, exists
        filters: list of {property, op, value} dicts
        spatial: {center_lat, center_lon, radius_m} or None
        dsn: PostgreSQL connection string

    Returns:
        dict with count/features/feature depending on operation
    """
    conn = psycopg2.connect(dsn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Build WHERE clause from filters
    where_parts = []
    params = []
    for f in (filters or []):
        prop, op, val = f["property"], f.get("op", "eq"), f["value"]
        if op == "eq":
            where_parts.append(f'LOWER("{prop}") = LOWER(%s)')
            params.append(str(val))
        elif op == "contains":
            where_parts.append(f'LOWER("{prop}") LIKE LOWER(%s)')
            params.append(f"%{val}%")
        elif op == "gte":
            where_parts.append(f'"{prop}" >= %s')
            params.append(val)
        elif op == "lte":
            where_parts.append(f'"{prop}" <= %s')
            params.append(val)

    # Add spatial constraint
    if spatial and spatial.get("center_lat"):
        where_parts.append(
            "ST_DWithin(geom::geography, "
            "ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)"
        )
        params.extend([spatial["center_lon"], spatial["center_lat"],
                       spatial.get("radius_m", 400)])

    where_clause = " AND ".join(where_parts) if where_parts else "TRUE"

    if operation in ("count", "list", "exists"):
        cur.execute(f'SELECT count(*) as cnt FROM "{dataset}" WHERE {where_clause}', params)
        count = cur.fetchone()["cnt"]

        cur.execute(
            f'SELECT *, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon '
            f'FROM "{dataset}" WHERE {where_clause} LIMIT 20', params
        )
        features = [
            {k: v for k, v in dict(r).items() if k != 'geom' and v is not None}
            for r in cur.fetchall()
        ]
        cur.close(); conn.close()
        return {"count": count, "features": features}

    elif operation == "nearest":
        lat, lon = spatial["center_lat"], spatial["center_lon"]
        radius = spatial.get("radius_m", 5000)
        cur.execute(f"""
            SELECT *,
                   ST_Y(ST_Centroid(geom)) as lat,
                   ST_X(ST_Centroid(geom)) as lon,
                   ST_Distance(geom::geography,
                     ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography)::int as distance_m
            FROM "{dataset}" WHERE {where_clause}
            AND ST_DWithin(geom::geography,
                  ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)
            ORDER BY geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geometry
            LIMIT 1
        """, [lon, lat] + params + [lon, lat, radius, lon, lat])
        row = cur.fetchone()
        cur.close(); conn.close()
        if row:
            feat = {k: v for k, v in dict(row).items() if k != 'geom' and v is not None}
            return {"feature": feat, "distance_m": feat.get("distance_m")}
        return {"feature": None, "distance_m": None}
