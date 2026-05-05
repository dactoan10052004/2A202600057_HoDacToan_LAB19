from __future__ import annotations

from typing import Any


ALIAS_MAP = {
    "Google LLC": "Google",
    "Alphabet Inc.": "Alphabet",
    "OpenAI, Inc.": "OpenAI",
    "OpenAI Global, LLC": "OpenAI",
    "Microsoft Corporation": "Microsoft",
    "Apple Inc.": "Apple",
    "Meta Platforms, Inc.": "Meta",
    "Meta Platforms": "Meta",
    "Amazon.com, Inc.": "Amazon",
    "Amazon (company)": "Amazon",
    "Nvidia Corporation": "Nvidia",
    "NVIDIA Corporation": "Nvidia",
    "YouTube, LLC": "YouTube",
    "YouTube LLC": "YouTube",
    "Alphabet Inc": "Alphabet",
    "OpenAI, Inc": "OpenAI",
    "OpenAI Global, LLC": "OpenAI",
    "Apple Inc": "Apple",
    "Meta Platforms, Inc": "Meta",
    "Amazon.com, Inc": "Amazon",
}


def normalize_entity(name: str) -> str:
    normalized = " ".join(str(name).strip().split())
    if normalized in ALIAS_MAP:
        return ALIAS_MAP[normalized]
    normalized = normalized.strip(" .;,")
    return ALIAS_MAP.get(normalized, normalized)


def normalize_predicate(predicate: str) -> str:
    return str(predicate).strip().upper().replace(" ", "_")


def deduplicate_triples(triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for triple in triples:
        subject = normalize_entity(triple.get("subject", ""))
        predicate = normalize_predicate(triple.get("predicate", ""))
        obj = normalize_entity(triple.get("object", ""))
        if not subject or not predicate or not obj:
            continue
        key = (subject.lower(), predicate, obj.lower())
        source_ids = triple.get("source_chunk_ids") or triple.get("source_chunk_id") or []
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        candidate = {
            **triple,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": float(triple.get("confidence", 0.0)),
            "source_chunk_ids": list(dict.fromkeys(source_ids)),
        }
        candidate.pop("source_chunk_id", None)
        existing = merged.get(key)
        if existing is None:
            merged[key] = candidate
            continue
        existing_sources = list(existing.get("source_chunk_ids", []))
        if candidate["confidence"] > float(existing.get("confidence", 0.0)):
            method = candidate.get("method", existing.get("method"))
            existing.update(candidate)
            existing["method"] = method
        existing["source_chunk_ids"] = list(
            dict.fromkeys(existing_sources + candidate["source_chunk_ids"])
        )
    return list(merged.values())
