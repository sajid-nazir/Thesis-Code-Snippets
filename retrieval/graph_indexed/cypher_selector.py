"""Cypher-based dataset selector for Graph-Indexed RAG.

Uses an LLM to generate Cypher queries against the Neo4j graph,
then executes them to identify relevant datasets for a given question.

The `run_cypher` callable and `schema_description` string are injected so the
module stays independent of any particular Neo4j client or configuration.
"""

import logging
import re
from typing import Callable

import anthropic

logger = logging.getLogger(__name__)

CYPHER_MAX_RETRIES = 2
MAX_DATASETS_PER_QUERY = 3

SYSTEM_PROMPT = """You generate Cypher queries for a Neo4j graph of Danish geospatial datasets. Given the schema and a question, write a MATCH query that returns the relevant Dataset nodes.

RULES:
1. Return ONLY the Cypher query. No explanation, no markdown, no code fences.
2. Always RETURN d.key (the dataset key is required for downstream processing).
3. Use domain_tags, title, description, and property_keys to determine relevance.
4. Consider synonyms: "play area" = playground, "bike rack" = bicycle parking, "charge" = EV charging, "water bottle" = drinking water, etc.
5. For location-specific questions, filter by Municipality or Region relationships.
6. Prefer CONTAINS on toLower() for text matching to be flexible.
7. Return ONLY 1 dataset in most cases. Return 2-3 only when the question explicitly combines multiple data types.
8. ORDER results by specificity: prefer datasets whose title or description most closely matches the question topic. Put the most relevant dataset FIRST.
9. When multiple datasets match (e.g., both 'parking' tagged), prefer the one whose title/description best matches the question's specific topic (bicycle vs car, etc.).
"""


def _extract_cypher(text: str) -> str:
    """Extract a Cypher query from the LLM response, handling code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text


def _extract_dataset_keys(rows: list[dict]) -> list[str]:
    """Extract dataset keys from query result rows, tolerating field-name variants."""
    keys = []
    for row in rows:
        for field in ("d.key", "key", "dataset_key", "d"):
            if field in row:
                val = row[field]
                if isinstance(val, str):
                    keys.append(val)
                    break
                elif isinstance(val, dict) and "key" in val:
                    keys.append(val["key"])
                    break

    seen = set()
    unique_keys = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)
    return unique_keys[:MAX_DATASETS_PER_QUERY]


def _fallback_text_search(question: str, run_cypher: Callable[[str], list[dict]]) -> list[str]:
    """Fallback: search datasets by text CONTAINS on title and description."""
    stop_words = {
        "i", "the", "a", "an", "is", "are", "there", "any", "my", "me",
        "can", "do", "where", "how", "many", "what", "which", "near",
        "within", "from", "to", "in", "on", "at", "of", "for", "and",
        "or", "with", "have", "has", "had", "be", "was", "were", "been",
        "this", "that", "these", "those", "it", "its", "im", "ive",
        "about", "around", "close", "walking", "distance", "nearby",
    }
    words = re.findall(r"[a-zA-ZæøåÆØÅ]+", question.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    if not keywords:
        return []

    conditions = []
    for kw in keywords[:5]:
        conditions.append(f"toLower(d.title) CONTAINS '{kw}'")
        conditions.append(f"toLower(d.description) CONTAINS '{kw}'")
    where_clause = " OR ".join(conditions)
    cypher = f"MATCH (d:Dataset) WHERE {where_clause} RETURN d.key LIMIT 3"

    try:
        rows = run_cypher(cypher)
        return _extract_dataset_keys(rows)
    except RuntimeError as exc:
        logger.error("Fallback search failed: %s", exc)
        return []


def select_datasets_via_cypher(
    question: str,
    client: anthropic.Anthropic,
    model: str,
    run_cypher: Callable[[str], list[dict]],
    schema_description: str,
    max_tokens: int = 2048,
) -> dict:
    """Select relevant datasets by generating and executing Cypher queries.

    Args:
        question: The user's natural-language question.
        client: Anthropic client instance.
        model: Model identifier string.
        run_cypher: Callable that executes a Cypher string and returns result rows.
        schema_description: Text description of the graph schema for the prompt.
        max_tokens: Generation budget.

    Returns:
        Dict with "datasets" list of keys, optionally "cypher", "retries", "reason".
    """
    user_message = (
        f"Graph Schema:\n{schema_description}\n\n"
        f"Question: {question}\n\n"
        f"Write a Cypher query to find the relevant dataset(s)."
    )

    cypher_query = None
    retries = 0
    last_error = None

    for attempt in range(CYPHER_MAX_RETRIES + 1):
        try:
            messages = [{"role": "user", "content": user_message}]
            if last_error and attempt > 0:
                messages.append({"role": "assistant", "content": cypher_query or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        f"The above Cypher query produced an error:\n{last_error}\n\n"
                        f"Please fix the query and return ONLY the corrected Cypher."
                    ),
                })

            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            response_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    response_text += block.text

            cypher_query = _extract_cypher(response_text)
            rows = run_cypher(cypher_query)
            dataset_keys = _extract_dataset_keys(rows)

            if dataset_keys:
                return {"datasets": dataset_keys, "cypher": cypher_query, "retries": retries}

            fallback_keys = _fallback_text_search(question, run_cypher)
            if fallback_keys:
                return {
                    "datasets": fallback_keys, "cypher": cypher_query,
                    "retries": retries, "fallback_used": True,
                }
            return {
                "datasets": [], "cypher": cypher_query, "retries": retries,
                "reason": "UNANSWERABLE: No matching datasets found in graph",
            }

        except RuntimeError as exc:
            last_error = str(exc)
            retries += 1
            logger.warning("Cypher execution error (attempt %d): %s", attempt + 1, last_error)
            if attempt >= CYPHER_MAX_RETRIES:
                fallback_keys = _fallback_text_search(question, run_cypher)
                if fallback_keys:
                    return {
                        "datasets": fallback_keys, "cypher": cypher_query,
                        "retries": retries, "fallback_used": True,
                    }
                return {
                    "datasets": [], "cypher": cypher_query, "retries": retries,
                    "reason": f"UNANSWERABLE: Cypher execution failed after {retries} retries: {last_error}",
                }
        except anthropic.APIError as exc:
            logger.error("Anthropic API error (attempt %d): %s", attempt + 1, exc)
            retries += 1
            if attempt >= CYPHER_MAX_RETRIES:
                return {
                    "datasets": [], "retries": retries,
                    "reason": f"UNANSWERABLE: LLM API error: {exc}",
                }
        except Exception as exc:
            logger.error("Unexpected error (attempt %d): %s", attempt + 1, exc)
            return {
                "datasets": [], "retries": retries,
                "reason": f"UNANSWERABLE: Unexpected error: {exc}",
            }

    return {"datasets": [], "retries": retries, "reason": "UNANSWERABLE: All attempts exhausted"}
