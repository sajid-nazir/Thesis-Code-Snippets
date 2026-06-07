"""Dataset selector for the PageIndex RAG pipeline.

Uses the collection tree + LLM to select 1-3 relevant datasets for a given question.
"""

import json
import logging
import re

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a geospatial dataset selector for Danish municipal open data.

Given a hierarchical collection tree of available datasets and a user question, select 1-3 relevant dataset keys that are most likely to contain the answer.

RULES:
1. Return ONLY valid JSON. No markdown, no explanation, no code fences.
2. Return format: {"datasets": ["key1", "key2"]} with 1-3 dataset keys.
3. If the question cannot be answered by ANY available dataset, return: {"datasets": [], "reason": "UNANSWERABLE: <brief explanation>"}
4. Only use dataset keys that appear in the collection tree (inside square brackets).
5. Use domain_tags, description, and fields to decide relevance.
6. For questions mentioning specific areas or municipalities, prefer datasets from that municipality.
7. Prefer datasets with more features when multiple datasets could answer the question.
8. Consider synonyms: "play area" = playground, "bike rack" = bicycle parking, etc.
"""


def _extract_known_keys(collection_tree_text: str) -> set[str]:
    """Extract all dataset keys from the rendered collection tree text."""
    return set(re.findall(r"\[([^\]]+)\]", collection_tree_text))


def _parse_response(text: str, known_keys: set[str]) -> dict | None:
    """Parse LLM response as JSON, with a regex fallback for known keys."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            datasets = result.get("datasets", [])
            valid_datasets = [k for k in datasets if k in known_keys]
            if valid_datasets or "reason" in result:
                result["datasets"] = valid_datasets
                return result
    except json.JSONDecodeError:
        pass

    quoted_strings = re.findall(r'"([^"]+)"', text)
    matched_keys = [s for s in quoted_strings if s in known_keys]
    if matched_keys:
        return {"datasets": matched_keys[:3]}

    if "UNANSWERABLE" in text.upper():
        return {"datasets": [], "reason": "UNANSWERABLE: Could not determine relevant dataset"}

    return None


def select_datasets(
    question: str,
    collection_tree_text: str,
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int = 2048,
) -> dict:
    """Select relevant datasets using the LLM.

    Args:
        question: The user's natural-language question.
        collection_tree_text: Rendered text of the collection tree.
        client: Anthropic client instance.
        model: Model identifier string.
        max_tokens: Generation budget.

    Returns:
        Dict with "datasets" list of keys, or "datasets": [] + "reason" if unanswerable.
    """
    known_keys = _extract_known_keys(collection_tree_text)
    user_message = (
        f"Available datasets:\n\n{collection_tree_text}\n\n"
        f"Question: {question}\n\n"
        f"Select 1-3 relevant dataset keys."
    )

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

            result = _parse_response(response_text, known_keys)
            if result is not None:
                return result

            if attempt == 0:
                user_message += (
                    "\n\nIMPORTANT: Your previous response was not valid JSON. "
                    "Return ONLY a JSON object like {\"datasets\": [\"key1\"]}. "
                    "Use ONLY keys from the square brackets above."
                )
        except anthropic.APIError as exc:
            logger.error("Anthropic API error (attempt %d): %s", attempt + 1, exc)
            if attempt == 0:
                continue
            break
        except Exception as exc:
            logger.error("Unexpected error in select_datasets (attempt %d): %s", attempt + 1, exc)
            break

    return {"datasets": [], "reason": "UNANSWERABLE: Failed to parse LLM response"}
