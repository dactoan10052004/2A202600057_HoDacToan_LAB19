from src.graph_builder import build_graph
from src.graph_query import GraphRAGEngine


def test_graph_query_collects_one_hop_evidence() -> None:
    graph = build_graph(
        [
            {
                "subject": "OpenAI",
                "predicate": "FOUNDED_BY",
                "object": "Sam Altman",
                "confidence": 0.9,
                "source_chunk_id": "chunk_000001",
            },
            {
                "subject": "Microsoft",
                "predicate": "INVESTED_IN",
                "object": "OpenAI",
                "confidence": 0.9,
                "source_chunk_id": "chunk_000002",
            },
        ]
    )
    engine = GraphRAGEngine(graph, chunks=[])
    result = engine.answer("Tell me about OpenAI", max_hops=1)
    evidence = result["evidence_triples"]
    assert len(evidence) == 2
    assert ("OpenAI", "FOUNDED_BY", "Sam Altman") in {
        (triple["subject"], triple["predicate"], triple["object"]) for triple in evidence
    }
    assert ("Microsoft", "INVESTED_IN", "OpenAI") in {
        (triple["subject"], triple["predicate"], triple["object"]) for triple in evidence
    }


def test_graph_query_ranks_question_relevant_predicates_first() -> None:
    graph = build_graph(
        [
            {
                "subject": "Google",
                "predicate": "INDUSTRY",
                "object": "technology",
                "confidence": 0.9,
                "source_chunk_id": "chunk_000001",
            },
            {
                "subject": "Google",
                "predicate": "FOUNDED_BY",
                "object": "Larry Page",
                "confidence": 0.9,
                "source_chunk_id": "chunk_000002",
            },
        ]
    )
    engine = GraphRAGEngine(graph, chunks=[])
    evidence = engine.collect_neighborhood("Google", max_hops=1)
    ranked = engine.rank_evidence("Who founded Google?", evidence)
    assert ranked[0]["predicate"] == "FOUNDED_BY"
