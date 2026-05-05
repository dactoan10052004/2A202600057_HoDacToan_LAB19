from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import _bootstrap  # noqa: F401
from src.config import CONFIG
from src.corpus_loader import ensure_dirs, load_jsonl
from src.llm_client import OpenAIClient
from src.utils import setup_logging, write_json


FALLBACK_QUESTIONS = [
    {"id": "q001", "question": "Who founded OpenAI?", "expected_answer": "Sam Altman", "type": "single-hop"},
    {"id": "q002", "question": "When was Google founded?", "expected_answer": "1998", "type": "single-hop"},
    {"id": "q003", "question": "Where is Microsoft headquartered?", "expected_answer": "Redmond", "type": "single-hop"},
    {"id": "q004", "question": "Who founded Apple?", "expected_answer": "Steve Jobs", "type": "single-hop"},
    {"id": "q005", "question": "Which company acquired DeepMind?", "expected_answer": "Google", "type": "single-hop"},
    {"id": "q006", "question": "Who founded Amazon?", "expected_answer": "Jeff Bezos", "type": "single-hop"},
    {"id": "q007", "question": "Who founded Nvidia?", "expected_answer": "Jensen Huang", "type": "single-hop"},
    {"id": "q008", "question": "Where is Apple headquartered?", "expected_answer": "Cupertino", "type": "single-hop"},
    {"id": "q009", "question": "Who founded Microsoft?", "expected_answer": "Bill Gates", "type": "single-hop"},
    {"id": "q010", "question": "Who founded Facebook or Meta?", "expected_answer": "Mark Zuckerberg", "type": "single-hop"},
    {"id": "q011", "question": "Who founded the company that Microsoft invested in?", "expected_answer": "Sam Altman", "type": "multi-hop"},
    {"id": "q012", "question": "Which company owns YouTube and who founded that company?", "expected_answer": "Larry Page", "type": "multi-hop"},
    {"id": "q013", "question": "Which parent is connected to DeepMind and who founded that parent?", "expected_answer": "Larry Page", "type": "multi-hop"},
    {"id": "q014", "question": "Who founded the company that owns LinkedIn?", "expected_answer": "Bill Gates", "type": "multi-hop"},
    {"id": "q015", "question": "Who founded the company that owns GitHub?", "expected_answer": "Bill Gates", "type": "multi-hop"},
    {"id": "q016", "question": "Who founded the parent company of Facebook?", "expected_answer": "Mark Zuckerberg", "type": "multi-hop"},
    {"id": "q017", "question": "Where is the company that owns YouTube headquartered?", "expected_answer": "Mountain View", "type": "multi-hop"},
    {"id": "q018", "question": "Who founded the company that acquired DeepMind?", "expected_answer": "Sergey Brin", "type": "multi-hop"},
    {"id": "q019", "question": "What industry is the parent company of Google associated with?", "expected_answer": "technology", "type": "multi-hop"},
    {"id": "q020", "question": "Who founded the company that developed Android?", "expected_answer": "Larry Page", "type": "multi-hop"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create benchmark questions from extracted triples.")
    parser.add_argument("--single", type=int, default=10, help="Number of single-hop questions.")
    parser.add_argument("--multi", type=int, default=10, help="Number of multi-hop questions.")
    parser.add_argument("--triples", default=str(CONFIG.triples_path), help="Input triples JSONL path.")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic generator only.")
    return parser.parse_args()


def _clean(value: Any) -> str:
    return str(value).strip()


def _is_answerable(value: str) -> bool:
    if not value or len(value) < 2 or len(value) > 80:
        return False
    bad_fragments = ("full list", "see ", "citation", "unknown", "various")
    return not any(fragment in value.lower() for fragment in bad_fragments)


def _is_entity_like(value: str) -> bool:
    if not _is_answerable(value):
        return False
    if any(marker in value for marker in ("(", "%", ")", "{", "}")):
        return False
    lowered = value.lower()
    generic = {
        "nonprofit foundation",
        "technology",
        "artificial intelligence",
        "software",
        "semiconductor",
        "electronics",
        "cloud computing",
    }
    if lowered in generic:
        return False
    return any(char.isupper() for char in value)


def _short_expected(question: str, expected: str) -> str:
    if question.lower().startswith("when was"):
        import re

        match = re.search(r"\b(18|19|20)\d{2}\b", expected)
        if match:
            return match.group(0)
    return expected


def _dedupe_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = row["question"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_single_hop_questions(triples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    priority = {
        "FOUNDED_BY": lambda s, o: f"Who founded {s}?",
        "FOUNDED_IN": lambda s, o: f"When was {s} founded?",
        "HEADQUARTERED_IN": lambda s, o: f"Where is {s} headquartered?",
        "ACQUIRED_BY": lambda s, o: f"Which company acquired {s}?",
        "PARENT_COMPANY": lambda s, o: f"What is the parent company of {s}?",
        "INDUSTRY": lambda s, o: f"What industry is {s} associated with?",
    }
    rows: list[dict[str, Any]] = []
    for predicate in priority:
        for triple in triples:
            if triple.get("predicate") != predicate:
                continue
            subject = _clean(triple.get("subject"))
            obj = _clean(triple.get("object"))
            if not _is_answerable(subject) or not _is_answerable(obj):
                continue
            rows.append(
                {
                    "question": priority[predicate](subject, obj),
                    "expected_answer": _short_expected(priority[predicate](subject, obj), obj),
                    "type": "single-hop",
                    "source": "generated_from_triples",
                }
            )
            if len(_dedupe_questions(rows)) >= limit:
                return _dedupe_questions(rows)[:limit]
    return _dedupe_questions(rows)[:limit]


def build_multi_hop_questions(triples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for triple in triples:
        outgoing[_clean(triple.get("subject"))].append(triple)

    rows: list[dict[str, Any]] = []
    answer_predicates = {"FOUNDED_BY", "FOUNDED_IN", "HEADQUARTERED_IN", "INDUSTRY"}
    bridge_counts: dict[str, int] = defaultdict(int)

    def add_questions(source: str, bridge: str, relation_text: str, path_predicate: str) -> None:
        if not _is_entity_like(source) or not _is_entity_like(bridge):
            return
        for second in outgoing.get(bridge, []):
            second_predicate = _clean(second.get("predicate"))
            answer = _clean(second.get("object"))
            if second_predicate not in answer_predicates or not _is_answerable(answer):
                continue
            if second_predicate == "FOUNDED_BY":
                question = f"Who founded the company that {relation_text}?"
            elif second_predicate == "FOUNDED_IN":
                question = f"When was the company that {relation_text} founded?"
            elif second_predicate == "HEADQUARTERED_IN":
                question = f"Where is the company that {relation_text} headquartered?"
            else:
                question = f"What industry is the company that {relation_text} associated with?"
            rows.append(
                {
                    "question": question,
                    "expected_answer": _short_expected(question, answer),
                    "type": "multi-hop",
                    "source": "generated_from_triples",
                    "path_hint": f"{source} -[{path_predicate}]-> {bridge} -[{second_predicate}]-> {answer}",
                    "bridge_key": f"{source}->{bridge}",
                }
            )

    seen_paths: set[tuple[str, str, str]] = set()
    for first in triples:
        first_predicate = _clean(first.get("predicate"))
        source = _clean(first.get("subject"))
        target = _clean(first.get("object"))
        path_key = (source, first_predicate, target)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)

        if first_predicate == "INVESTED_IN":
            add_questions(source, target, f"{source} invested in", first_predicate)
        elif first_predicate == "OWNS":
            add_questions(source, target, f"{source} owns", first_predicate)

    diversified: list[dict[str, Any]] = []
    for row in _dedupe_questions(rows):
        bridge_key = str(row.pop("bridge_key", ""))
        if bridge_counts[bridge_key] >= 2:
            continue
        bridge_counts[bridge_key] += 1
        diversified.append(row)
        if len(diversified) >= limit:
            break
    return diversified


def create_questions(triples: list[dict[str, Any]], single_count: int, multi_count: int) -> list[dict[str, Any]]:
    singles = build_single_hop_questions(triples, single_count)
    multis = build_multi_hop_questions(triples, multi_count)

    fallback_singles = [row for row in FALLBACK_QUESTIONS if row["type"] == "single-hop"]
    fallback_multis = [row for row in FALLBACK_QUESTIONS if row["type"] == "multi-hop"]
    singles = _dedupe_questions(singles + fallback_singles)[:single_count]
    multis = _dedupe_questions(multis + fallback_multis)[:multi_count]

    questions = singles + multis
    for index, row in enumerate(questions, start=1):
        row["id"] = f"q{index:03d}"
    return questions


def _candidate_brief(rows: list[dict[str, Any]], limit: int = 40) -> str:
    lines = []
    for row in rows[:limit]:
        extra = f" | path: {row.get('path_hint')}" if row.get("path_hint") else ""
        lines.append(
            f"- type={row['type']} | q={row['question']} | expected={row['expected_answer']}{extra}"
        )
    return "\n".join(lines)


def create_questions_with_llm(
    triples: list[dict[str, Any]],
    single_count: int,
    multi_count: int,
    client: OpenAIClient,
) -> list[dict[str, Any]]:
    single_candidates = build_single_hop_questions(triples, max(single_count * 3, 20))
    multi_candidates = build_multi_hop_questions(triples, max(multi_count * 4, 30))
    deterministic = create_questions(triples, single_count, multi_count)
    candidates = _dedupe_questions(single_candidates + multi_candidates + deterministic)
    prompt = f"""Bạn là evaluator cho bài lab GraphRAG.
Hãy tạo benchmark gồm đúng {single_count} câu single-hop và đúng {multi_count} câu multi-hop dựa trên candidates lấy từ knowledge graph.

Yêu cầu:
- Trả về JSON hợp lệ, không markdown.
- Câu hỏi bằng tiếng Anh để tương thích corpus.
- expected_answer phải là một chuỗi ngắn xuất hiện trong evidence/candidate.
- Single-hop chỉ cần một triple.
- Multi-hop phải cần ít nhất 2 cạnh/triples, ưu tiên câu mà Flat RAG dễ thiếu một mảnh evidence.
- Không tạo câu hỏi ngoài dữ liệu candidates.
- Ưu tiên đa dạng công ty/path; không lấy nhiều hơn 2 câu cho cùng một path.
- Tuyệt đối tránh câu mơ hồ kiểu "connected through" hoặc nhắc tên predicate như SUBSIDIARY_OF/ACQUIRED_BY.
- Hãy viết tự nhiên, ví dụ "Who founded the company that owns YouTube?" hoặc "Who founded the company that Microsoft invested in?"

Schema:
{{
  "questions": [
    {{
      "id": "q001",
      "question": "...",
      "expected_answer": "...",
      "type": "single-hop",
      "notes": "short evidence/path"
    }}
  ]
}}

Candidates:
{_candidate_brief(candidates)}
"""
    payload = client.generate_json(prompt, CONFIG.answer_model)
    rows = payload.get("questions", [])
    if not isinstance(rows, list):
        raise ValueError("LLM benchmark response missing questions list")

    cleaned: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        qtype = row.get("type")
        question = _clean(row.get("question"))
        expected = _clean(row.get("expected_answer"))
        expected = _short_expected(question, expected)
        if qtype not in {"single-hop", "multi-hop"} or not question or not _is_answerable(expected):
            continue
        cleaned.append(
            {
                "question": question,
                "expected_answer": expected,
                "type": qtype,
                "source": "llm_generated_from_triples",
                "notes": _clean(row.get("notes")),
            }
        )

    singles = [row for row in _dedupe_questions(cleaned) if row["type"] == "single-hop"][:single_count]
    multis = [row for row in _dedupe_questions(cleaned) if row["type"] == "multi-hop"][:multi_count]
    if len(singles) < single_count or len(multis) < multi_count:
        fallback = create_questions(triples, single_count, multi_count)
        singles = _dedupe_questions(singles + [row for row in fallback if row["type"] == "single-hop"])[:single_count]
        multis = _dedupe_questions(multis + [row for row in fallback if row["type"] == "multi-hop"])[:multi_count]

    questions = singles + multis
    for index, row in enumerate(questions, start=1):
        row["id"] = f"q{index:03d}"
    return questions


def main() -> None:
    setup_logging()
    ensure_dirs()
    args = parse_args()
    triples = load_jsonl(args.triples)
    if not args.no_llm and CONFIG.openai_api_key:
        try:
            questions = create_questions_with_llm(
                triples,
                args.single,
                args.multi,
                OpenAIClient(CONFIG.openai_api_key),
            )
        except Exception as exc:
            print(f"LLM benchmark generation failed, using deterministic fallback: {exc}")
            questions = create_questions(triples, args.single, args.multi)
    else:
        questions = create_questions(triples, args.single, args.multi)
    write_json(questions, CONFIG.benchmark_questions_path)
    summary = {
        "questions": len(questions),
        "single-hop": sum(1 for row in questions if row["type"] == "single-hop"),
        "multi-hop": sum(1 for row in questions if row["type"] == "multi-hop"),
        "path": str(CONFIG.benchmark_questions_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
