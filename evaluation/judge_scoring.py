"""Evaluation scoring using structured LLM output.

Uses tool_use with forced tool_choice to guarantee a valid 0-3 integer score
for each question-answer pair. No parsing ambiguity.
"""

import json


JUDGE_SYSTEM = """You are evaluating a geospatial question-answering system.
Score the generated answer compared to the expected answer:
- 3: Correct — contains the exact expected information
- 2: Close — right approach, slightly off (e.g., off by 1-2 for numeric)
- 1: Partial — shows understanding but notably wrong
- 0: Wrong — completely wrong, irrelevant, or empty

For numeric answers: exact match = 3, within 10% = 2, within 25% = 1, else = 0.
For NOT_ANSWERABLE: correctly refusing = 3, answering with wrong data = 0."""

JUDGE_TOOL = {
    "name": "submit_score",
    "description": "Submit evaluation score",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "enum": [0, 1, 2, 3]},
            "reason": {"type": "string", "description": "Brief explanation"}
        },
        "required": ["score", "reason"]
    }
}


def score_answer(question, expected, generated, client, model):
    """Score a single answer using structured LLM output.

    Args:
        question: the user's question
        expected: the ground truth answer
        generated: the system's generated answer
        client: Anthropic client instance
        model: model identifier string

    Returns:
        (score, reason) tuple where score is 0-3
    """
    if not generated or generated.strip() == "":
        return 0, "Empty answer"

    response = client.messages.create(
        model=model,
        max_tokens=200,
        system=JUDGE_SYSTEM,
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "submit_score"},
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Expected answer: {expected}\n"
                f"Generated answer: {generated[:400]}\n\n"
                f"Evaluate."
            )
        }],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input["score"], block.input.get("reason", "")

    return 0, "No structured output returned"


def evaluate_system(results, ground_truth, client, model):
    """Evaluate all results for a system.

    Args:
        results: list of {id, final_answer, ...} dicts
        ground_truth: list of {id, answer_exact, question, ...} dicts
        client: Anthropic client
        model: model identifier

    Returns:
        list of {id, score, reason} dicts
    """
    gt_by_id = {q["id"]: q for q in ground_truth}
    scores = []

    for r in results:
        q = gt_by_id[r["id"]]
        score, reason = score_answer(
            q["question"], str(q["answer_exact"]),
            r.get("final_answer", ""), client, model
        )
        scores.append({"id": r["id"], "score": score, "reason": reason})

    return scores
