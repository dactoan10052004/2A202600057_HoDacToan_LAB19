from __future__ import annotations

from typing import Any

import pandas as pd

from src.cost_tracker import extract_token_usage


def contains_answer(prediction: str, expected: str | list[str]) -> bool:
    prediction_lower = prediction.lower()
    candidates = expected if isinstance(expected, list) else [expected]
    return any(str(candidate).lower() in prediction_lower for candidate in candidates)


def evaluate_row(
    question_obj: dict[str, Any],
    flat_result: dict[str, Any],
    graph_result: dict[str, Any],
) -> dict[str, Any]:
    expected = question_obj.get("expected_answer", "")
    flat_usage = extract_token_usage(flat_result)
    graph_usage = extract_token_usage(graph_result)
    return {
        "question_id": question_obj.get("id", ""),
        "question": question_obj.get("question", ""),
        "type": question_obj.get("type", ""),
        "expected_answer": expected if isinstance(expected, str) else "; ".join(expected),
        "flat_rag_answer": flat_result.get("answer", ""),
        "graphrag_answer": graph_result.get("answer", ""),
        "flat_rag_correct": contains_answer(flat_result.get("answer", ""), expected),
        "graphrag_correct": contains_answer(graph_result.get("answer", ""), expected),
        "flat_rag_latency": flat_result.get("latency_seconds", 0.0),
        "graphrag_latency": graph_result.get("latency_seconds", 0.0),
        "flat_rag_tokens": flat_usage["total_tokens"],
        "graphrag_tokens": graph_usage["total_tokens"],
        "graphrag_evidence_count": len(graph_result.get("evidence_triples", [])),
        "notes": "",
    }


def run_benchmark(
    questions: list[dict[str, Any]],
    flat_engine: Any,
    graph_engine: Any,
) -> pd.DataFrame:
    rows = []
    for question in questions:
        text = question["question"]
        flat_result = flat_engine.answer(text)
        graph_result = graph_engine.answer(text)
        rows.append(evaluate_row(question, flat_result, graph_result))
    return pd.DataFrame(rows)
