from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx


def build_graph(triples: list[dict[str, Any]]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for triple in triples:
        subject = str(triple.get("subject", "")).strip()
        obj = str(triple.get("object", "")).strip()
        predicate = str(triple.get("predicate", "")).strip()
        if not subject or not obj or not predicate:
            continue
        for node in (subject, obj):
            if graph.has_node(node):
                graph.nodes[node]["mentions"] += 1
            else:
                graph.add_node(node, name=node, mentions=1, type="Unknown")
        graph.add_edge(
            subject,
            obj,
            predicate=predicate,
            confidence=float(triple.get("confidence", 0.0)),
            source_chunk_ids=triple.get("source_chunk_ids")
            or [triple.get("source_chunk_id", "")],
            method=triple.get("method", "unknown"),
        )
    return graph


def compute_graph_stats(graph: nx.MultiDiGraph) -> dict[str, Any]:
    degree = dict(graph.degree())
    predicates = Counter(data.get("predicate", "UNKNOWN") for _, _, data in graph.edges(data=True))
    components = nx.number_weakly_connected_components(graph) if graph.number_of_nodes() else 0
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "top_degree_nodes": sorted(degree.items(), key=lambda item: item[1], reverse=True)[:10],
        "weakly_connected_components": components,
        "predicates_count": dict(predicates),
    }

