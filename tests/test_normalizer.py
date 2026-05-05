from src.normalizer import deduplicate_triples, normalize_entity


def test_normalize_entity_alias_map() -> None:
    assert normalize_entity("Google LLC") == "Google"
    assert normalize_entity("  Apple Inc. ") == "Apple"


def test_deduplicate_triples_keeps_highest_confidence_and_sources() -> None:
    triples = [
        {
            "subject": "Google LLC",
            "predicate": "acquired",
            "object": "DeepMind",
            "confidence": 0.6,
            "source_chunk_id": "chunk_000001",
        },
        {
            "subject": "Google",
            "predicate": "ACQUIRED",
            "object": "DeepMind",
            "confidence": 0.9,
            "source_chunk_id": "chunk_000002",
        },
    ]
    deduped = deduplicate_triples(triples)
    assert len(deduped) == 1
    assert deduped[0]["confidence"] == 0.9
    assert deduped[0]["source_chunk_ids"] == ["chunk_000001", "chunk_000002"]

