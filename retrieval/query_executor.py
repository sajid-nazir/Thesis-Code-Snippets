"""Query executor for structured geospatial queries.

Executes a query plan against GeoJSON collection files,
applying property filters and spatial constraints.
"""

import json
import math
from pathlib import Path


def haversine_m(lat1, lon1, lat2, lon2):
    """Compute Haversine distance between two WGS84 points in metres."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def apply_filter(value, op, filter_value):
    """Apply a single filter operation (eq, neq, contains, gte, lte, in)."""
    if value is None:
        return op == "eq" and filter_value is None
    if op == "eq":
        if isinstance(value, str) and isinstance(filter_value, str):
            return value.lower() == filter_value.lower()
        return value == filter_value
    elif op == "neq":
        return value != filter_value
    elif op == "contains":
        return str(filter_value).lower() in str(value).lower()
    elif op == "gte":
        return float(value) >= float(filter_value)
    elif op == "lte":
        return float(value) <= float(filter_value)
    elif op == "in":
        if isinstance(filter_value, list):
            return value in filter_value
        return value == filter_value
    return False


def execute_plan(plan, collections_dir):
    """Execute a query plan against a GeoJSON collection.

    Args:
        plan: dict with keys: operation, dataset, filters, spatial
        collections_dir: Path to directory containing collection JSON files

    Returns:
        dict with count/features/feature depending on operation type
    """
    operation = plan.get("operation", "list")
    dataset = plan.get("dataset", "")
    filters = plan.get("filters", [])
    spatial = plan.get("spatial", {})

    # Load features
    path = Path(collections_dir) / f"{dataset}.json"
    with open(path) as f:
        data = json.load(f)
    features = data.get("geojson", {}).get("features", [])

    # Apply property filters
    filtered = []
    for feat in features:
        props = feat.get("properties", {})
        if all(apply_filter(props.get(f["property"]), f["op"], f["value"]) for f in filters):
            filtered.append(feat)

    # Apply spatial filter
    center_lat = spatial.get("center_lat")
    center_lon = spatial.get("center_lon")
    radius_m = spatial.get("radius_m")

    if center_lat and center_lon and radius_m:
        spatial_results = []
        for feat in filtered:
            coords = feat.get("geometry", {}).get("coordinates", [])
            if isinstance(coords[0], list):
                coords = coords[0]  # Handle MultiPoint
            feat_lon, feat_lat = coords[0], coords[1]
            dist = haversine_m(center_lat, center_lon, feat_lat, feat_lon)
            if dist <= radius_m:
                spatial_results.append((feat, dist))
        spatial_results.sort(key=lambda x: x[1])
    else:
        spatial_results = [(f, None) for f in filtered]

    # Execute operation
    if operation == "count":
        return {"count": len(spatial_results), "features": spatial_results[:20]}
    elif operation == "nearest":
        if spatial_results:
            return {"feature": spatial_results[0][0], "distance_m": spatial_results[0][1]}
        return {"feature": None, "distance_m": None}
    elif operation == "list":
        return {"count": len(spatial_results), "features": spatial_results[:50]}
    elif operation == "exists":
        return {"exists": len(spatial_results) > 0, "count": len(spatial_results)}
