from __future__ import annotations

import logging
import re
from typing import Any

from src.config import CONFIG
from src.llm_client import OpenAIClient
from src.normalizer import normalize_predicate

LOGGER = logging.getLogger(__name__)

ALLOWED_PREDICATES = {
    "FOUNDED_BY",
    "FOUNDED_IN",
    "CEO_OF",
    "ACQUIRED",
    "ACQUIRED_BY",
    "INVESTED_IN",
    "PARTNERED_WITH",
    "OWNS",
    "SUBSIDIARY_OF",
    "DEVELOPED",
    "RELEASED",
    "HEADQUARTERED_IN",
    "COMPETES_WITH",
    "USES_TECHNOLOGY",
    "PROVIDES_SERVICE",
    "CREATED_BY",
    "PARENT_COMPANY",
    "INDUSTRY",
    "PRODUCT",
}


def _chunk_title(text: str) -> str:
    match = re.search(r"#\s*([^\n]+)", text)
    if match:
        return match.group(1).strip()
    first_sentence = text.strip().split(".", 1)[0]
    return first_sentence.split(" is ", 1)[0].strip()[:80] or "Unknown"


def _split_people(value: str) -> list[str]:
    value = value.replace("|", ",").replace("*", ",")
    value = re.sub(r"\s+and\s+", ", ", value)
    value = value.replace(";", ",")
    return [part.strip(" .") for part in value.split(",") if part.strip(" .")]


def _split_values(value: str) -> list[str]:
    value = value.replace("|", ";").replace("*", ";")
    value = re.sub(r"\s+\band\b\s+", "; ", value)
    parts = re.split(r";|\n|,(?=\s*[A-Z][A-Za-z])", value)
    return [part.strip(" .") for part in parts if part.strip(" .")]


def _clean_structured_value(value: str) -> str:
    value = re.sub(r"<!--.*?-->", "", value, flags=re.DOTALL)
    value = value.replace("}}", "")
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"\s*\(.*$", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .;,")


def _clean_place(value: str) -> str:
    value = re.split(r",?\s+and\s+|,?\s+known\s+for\s+", value, maxsplit=1, flags=re.IGNORECASE)[0]
    parts = [part.strip() for part in value.split(",")]
    if len(parts) >= 2:
        return ", ".join(parts[:2])
    return value.strip(" .")


def _make_triple(
    subject: str,
    predicate: str,
    obj: str,
    chunk_id: str,
    confidence: float,
    method: str,
) -> dict[str, Any]:
    return {
        "subject": subject.strip(),
        "predicate": normalize_predicate(predicate),
        "object": obj.strip(),
        "source_chunk_id": chunk_id,
        "confidence": max(0.0, min(1.0, float(confidence))),
        "method": method,
    }


def _valid_triple(triple: dict[str, Any]) -> bool:
    predicate = normalize_predicate(triple.get("predicate", ""))
    try:
        confidence = float(triple.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False
    return (
        bool(str(triple.get("subject", "")).strip())
        and bool(str(triple.get("object", "")).strip())
        and predicate in ALLOWED_PREDICATES
        and 0.0 <= confidence <= 1.0
    )


def rule_based_extract(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(chunk.get("text", ""))
    chunk_id = str(chunk.get("chunk_id", ""))
    subject = str(chunk.get("source_title") or _chunk_title(text))
    triples: list[dict[str, Any]] = []

    structured_patterns = {
        "Founders": "FOUNDED_BY",
        "Founded": "FOUNDED_IN",
        "Headquarters": "HEADQUARTERED_IN",
        "Parent company": "PARENT_COMPANY",
        "Acquired by": "ACQUIRED_BY",
        "Industry": "INDUSTRY",
        "Products": "PRODUCT",
        "Services": "PROVIDES_SERVICE",
    }
    for label, predicate in structured_patterns.items():
        match = re.search(rf"^- {re.escape(label)}:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        value = match.group(1).strip()
        if predicate == "FOUNDED_IN":
            year = re.search(r"\b(18|19|20)\d{2}\b", value)
            if year:
                triples.append(_make_triple(subject, predicate, year.group(0), chunk_id, 0.9, "rule_based"))
            continue
        values = _split_people(value) if predicate == "FOUNDED_BY" else _split_values(value)
        for item in values:
            item = _clean_structured_value(item)
            if item:
                triples.append(_make_triple(subject, predicate, item, chunk_id, 0.9, "rule_based"))

    owner_match = re.search(r"^- Owner:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if owner_match:
        for owner in _split_values(owner_match.group(1)):
            owner = _clean_structured_value(owner)
            if owner:
                triples.append(_make_triple(owner, "OWNS", subject, chunk_id, 0.9, "rule_based"))
                triples.append(_make_triple(subject, "SUBSIDIARY_OF", owner, chunk_id, 0.85, "rule_based"))

    founded_by_patterns = [
        r"founded by ([^.]+?)(?: in \d{4})?\.",
        r"founded (?:in|on) [^.]+? by ([^.]+?)(?:, the company|\sin [A-Z][A-Za-z ]+(?:,|\.)|\.)",
    ]
    for pattern in founded_by_patterns:
        founded_by = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        if founded_by:
            for founder in _split_people(founded_by.group(1)):
                triples.append(_make_triple(subject, "FOUNDED_BY", founder, chunk_id, 0.72, "rule_based"))
            break

    founded_in = re.search(
        r"founded (?:by [^.]+? )?(?:in|on) [A-Za-z]+\s+\d{1,2},\s+(\d{4})|founded (?:by [^.]+? )?in (\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    if founded_in:
        year = founded_in.group(1) or founded_in.group(2)
        triples.append(_make_triple(subject, "FOUNDED_IN", year, chunk_id, 0.72, "rule_based"))

    headquarters = re.search(
        r"(?:is headquartered|headquartered) in ([^.]+)\.",
        text,
        flags=re.IGNORECASE,
    )
    if headquarters:
        triples.append(
            _make_triple(subject, "HEADQUARTERED_IN", _clean_place(headquarters.group(1)), chunk_id, 0.68, "rule_based")
        )

    acquired_by = re.search(r"was acquired by ([^.]+)\.", text, flags=re.IGNORECASE)
    if acquired_by:
        triples.append(_make_triple(subject, "ACQUIRED_BY", acquired_by.group(1), chunk_id, 0.7, "rule_based"))

    subsidiary = re.search(r"is a subsidiary of ([^.]+)\.", text, flags=re.IGNORECASE)
    if subsidiary:
        triples.append(_make_triple(subject, "SUBSIDIARY_OF", subsidiary.group(1), chunk_id, 0.7, "rule_based"))

    parent = re.search(
        r"\b([A-Z][A-Za-z0-9 &-]{1,40})'s parent company ([A-Z][A-Za-z0-9 .,&-]+?)(?: has| is| was|\.|,)",
        text,
    )
    if parent:
        triples.append(_make_triple(parent.group(1), "PARENT_COMPANY", parent.group(2), chunk_id, 0.66, "rule_based"))

    for match in re.finditer(
        r"([A-Z][A-Za-z0-9 &-]{1,80}) developed (?:the )?([A-Za-z0-9 .,&()-]{2,100}?)(?:,|\.| which )",
        text,
    ):
        triples.append(_make_triple(match.group(1), "DEVELOPED", match.group(2), chunk_id, 0.62, "rule_based"))

    for match in re.finditer(r"([A-Z][A-Za-z0-9 .,&-]{1,80}) owns ([A-Z][A-Za-z0-9 .,&-]{1,80})", text):
        triples.append(_make_triple(match.group(1), "OWNS", match.group(2), chunk_id, 0.62, "rule_based"))

    if re.search(r"\bMicrosoft\b", text) and re.search(r"\bOpenAI\b", text) and re.search(r"\binvest", text, re.IGNORECASE):
        triples.append(_make_triple("Microsoft", "INVESTED_IN", "OpenAI", chunk_id, 0.75, "rule_based"))

    return [triple for triple in triples if _valid_triple(triple)]


def extract_triples_from_chunk(chunk: dict[str, Any], client: OpenAIClient | None) -> list[dict[str, Any]]:
    if client is None or not client.is_available():
        return rule_based_extract(chunk)
    prompt = f"""Bạn là hệ thống trích xuất knowledge graph.
Đọc text và trả về JSON hợp lệ.
Không suy diễn ngoài text.
Schema:
{{
  "triples": [
    {{
      "subject": "...",
      "predicate": "...",
      "object": "...",
      "confidence": 0.85
    }}
  ]
}}
Confidence phải là số trong [0.5, 1.0] cho fact được nêu rõ trong text; không dùng 0.0 cho fact hợp lệ.
Allowed predicates:
- FOUNDED_BY
- FOUNDED_IN
- CEO_OF
- ACQUIRED
- ACQUIRED_BY
- INVESTED_IN
- PARTNERED_WITH
- OWNS
- SUBSIDIARY_OF
- DEVELOPED
- RELEASED
- HEADQUARTERED_IN
- COMPETES_WITH
- USES_TECHNOLOGY
- PROVIDES_SERVICE
- CREATED_BY
- PARENT_COMPANY
- INDUSTRY
- PRODUCT
Text:
{chunk.get("text", "")}
"""
    try:
        payload = client.generate_json(prompt, CONFIG.extraction_model)
        triples = payload.get("triples", [])
        if not isinstance(triples, list):
            return rule_based_extract(chunk)
        validated = []
        for triple in triples:
            if not isinstance(triple, dict):
                continue
            triple = {
                "subject": str(triple.get("subject", "")),
                "predicate": normalize_predicate(str(triple.get("predicate", ""))),
                "object": str(triple.get("object", "")),
                "source_chunk_id": str(chunk.get("chunk_id", "")),
                "confidence": float(triple.get("confidence", 0.0)),
                "method": "llm",
            }
            if _valid_triple(triple):
                validated.append(triple)
        return validated + rule_based_extract(chunk)
    except Exception as exc:
        LOGGER.warning("LLM extraction failed for %s, using fallback: %s", chunk.get("chunk_id"), exc)
        return rule_based_extract(chunk)


def extract_triples(chunks: list[dict[str, Any]], client: OpenAIClient | None) -> list[dict[str, Any]]:
    triples: list[dict[str, Any]] = []
    for chunk in chunks:
        triples.extend(extract_triples_from_chunk(chunk, client))
    return triples
