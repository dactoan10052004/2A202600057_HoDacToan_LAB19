from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import networkx as nx

import _bootstrap  # noqa: F401
from src.config import CONFIG
from src.graph_store import load_graph
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize graph to PNG.")
    parser.add_argument("--graph", default=str(CONFIG.graph_path), help="Input graph pickle path.")
    parser.add_argument("--output", default=str(CONFIG.graph_png_path), help="Output PNG path.")
    parser.add_argument("--max-edges", type=int, default=80, help="Maximum edges to render.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    graph = load_graph(args.graph)
    if graph.number_of_edges() > args.max_edges:
        top_edges = sorted(
            graph.edges(keys=True, data=True),
            key=lambda edge: graph.degree(edge[0]) + graph.degree(edge[1]),
            reverse=True,
        )[: args.max_edges]
        view = nx.MultiDiGraph()
        for source, target, _key, data in top_edges:
            view.add_node(source, **graph.nodes[source])
            view.add_node(target, **graph.nodes[target])
            view.add_edge(source, target, **data)
    else:
        view = graph

    plt.figure(figsize=(18, 12))
    pos = nx.spring_layout(view, seed=42, k=0.8)
    degrees = dict(view.degree())
    node_sizes = [400 + degrees[node] * 90 for node in view.nodes]
    nx.draw_networkx_nodes(view, pos, node_size=node_sizes, node_color="#78A6C8", alpha=0.9)
    nx.draw_networkx_edges(view, pos, arrows=True, alpha=0.35, width=1.2, edge_color="#444444")
    nx.draw_networkx_labels(view, pos, font_size=8)
    labels = {
        (source, target): data.get("predicate", "")
        for source, target, data in view.edges(data=True)
    }
    nx.draw_networkx_edge_labels(view, pos, edge_labels=labels, font_size=6)
    plt.axis("off")
    output_path = CONFIG.graph_png_path if args.output == str(CONFIG.graph_png_path) else args.output
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    print(f"Saved graph image to {output_path}")


if __name__ == "__main__":
    main()

