"""Query planner for the geospatial RAG pipeline.

Uses an LLM to convert a natural language question + schema tree into
a structured query plan (JSON) that the executor can run.
"""

import json
import logging
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

# Distance defaults (metres) inferred from natural-language phrasing.
DISTANCE_DEFAULTS = {
    "walking": 1000,
    "nearby": 500,
    "close": 300,
    "very_close": 150,
}

SYSTEM_PROMPT = f"""You are a query planner for a geospatial database. Given a schema description and a user question, produce a JSON query plan.

RULES:
1. Return ONLY valid JSON. No markdown, no explanation, no code fences.
2. The JSON must have this structure:
   {{"operation": "<op>", "dataset": "<key>", "filters": [...], "spatial": {{...}}}}

3. Supported operations: count, list, nearest, exists
   - "count": How many features match?
   - "list": Which features match? Return them.
   - "nearest": Which single feature is closest to a point?
   - "exists": Do any features match? (yes/no question)

4. Supported filter operators: eq, neq, contains, gte, lte, in
   Each filter: {{"property": "<name>", "op": "<operator>", "value": <value>}}

5. Only use property names that exist in the provided schema.

6. For spatial queries, include:
   "spatial": {{"center_lat": <lat>, "center_lon": <lon>, "radius_m": <meters>}}

7. Distance inference from language:
   - "walking distance" -> {DISTANCE_DEFAULTS['walking']}m
   - "nearby" -> {DISTANCE_DEFAULTS['nearby']}m
   - "close to" -> {DISTANCE_DEFAULTS['close']}m
   - "very close" -> {DISTANCE_DEFAULTS['very_close']}m

8. If the question mentions a specific location but no coordinates are available, omit the spatial field.

9. For "how many" questions, use operation "count".
   For "is there" / "are there" questions, use operation "exists".
   For "which" / "what" / "list" questions, use operation "list".
   For "nearest" / "closest" questions, use operation "nearest".
"""


def _build_schema_summary(schema_tree: dict) -> str:
    """Build a concise text summary of the schema tree for the LLM prompt."""
    title = schema_tree.get("title", "Unknown")
    dataset_key = schema_tree.get("dataset_key", "unknown")
    feature_count = schema_tree.get("feature_count", 0)
    properties = schema_tree.get("properties", {})

    lines = [
        f"Dataset: {title}",
        f"Key: {dataset_key}",
        f"Features: {feature_count}",
        "",
        "Properties:",
    ]

    for prop_name, stats in properties.items():
        prop_type = stats.get("type", "unknown")
        null_rate = stats.get("null_rate", 0)
        unique = stats.get("unique_count", 0)
        top = stats.get("top_values", [])

        line = f"  - {prop_name} ({prop_type})"
        if null_rate > 0.5:
            line += f" [null_rate={null_rate:.0%}]"
        if unique > 0:
            line += f" [{unique} unique]"
        if top and unique <= 20:
            vals = [str(t["value"]) for t in top[:5]]
            line += f' values: {", ".join(vals)}'
        if "min" in stats and "max" in stats:
            line += f" range: [{stats['min']}, {stats['max']}]"

        lines.append(line)

    return "\n".join(lines)


def _parse_plan_response(text: str) -> Optional[dict]:
    """Parse the LLM response into a query plan dict, tolerating code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        plan = json.loads(text)
        if isinstance(plan, dict):
            return plan
        logger.warning("LLM returned non-dict JSON: %s", type(plan))
        return None
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM response as JSON: %s", exc)
        return None


def _default_plan(dataset_key: str) -> dict:
    """Return a safe default plan when LLM parsing fails."""
    return {"operation": "list", "dataset": dataset_key, "filters": [], "spatial": {}}


def plan_query(
    question: str,
    schema_tree: dict,
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int = 2048,
    spatial_context: Optional[dict] = None,
) -> dict:
    """Generate a query plan from a natural-language question using the LLM.

    Args:
        question: The user's natural-language question.
        schema_tree: Schema tree dict for the target dataset.
        client: Anthropic client instance.
        model: Model identifier string.
        max_tokens: Generation budget.
        spatial_context: Optional dict with 'center_lat', 'center_lon', 'radius_m'.

    Returns:
        Query plan dict with keys: operation, dataset, filters, spatial.
    """
    dataset_key = schema_tree.get("dataset_key", "unknown")
    schema_summary = _build_schema_summary(schema_tree)

    user_parts = [f"Schema:\n{schema_summary}", f"\nQuestion: {question}"]
    if spatial_context:
        user_parts.append(
            f"\nSpatial context (use these coordinates): "
            f"center_lat={spatial_context.get('center_lat')}, "
            f"center_lon={spatial_context.get('center_lon')}, "
            f"radius_m={spatial_context.get('radius_m')}"
        )
    user_message = "\n".join(user_parts)

    for attempt in range(2):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            response_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    response_text += block.text

            plan = _parse_plan_response(response_text)
            if plan is not None:
                plan.setdefault("operation", "list")
                plan.setdefault("dataset", dataset_key)
                plan.setdefault("filters", [])
                plan.setdefault("spatial", {})
                if spatial_context and not plan.get("spatial"):
                    plan["spatial"] = spatial_context
                return plan

            if attempt == 0:
                user_message += (
                    "\n\nIMPORTANT: Your previous response was not valid JSON. "
                    "Return ONLY a JSON object, nothing else."
                )
        except anthropic.APIError as exc:
            logger.error("Anthropic API error (attempt %d): %s", attempt + 1, exc)
            if attempt == 0:
                continue
            break
        except Exception as exc:
            logger.error("Unexpected error in plan_query (attempt %d): %s", attempt + 1, exc)
            break

    return _default_plan(dataset_key)
