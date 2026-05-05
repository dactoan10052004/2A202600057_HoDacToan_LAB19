from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import networkx as nx

from src.utils import write_json


def save_graph(graph: nx.MultiDiGraph, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(graph, handle)


def load_graph(path: str | Path) -> nx.MultiDiGraph:
    with Path(path).open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError(f"Expected nx.MultiDiGraph, got {type(graph)!r}")
    return graph


def save_graph_stats(stats: dict[str, Any], path: str | Path) -> None:
    write_json(stats, path)

