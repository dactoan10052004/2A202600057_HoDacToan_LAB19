from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from src.config import CONFIG
from src.corpus_loader import load_jsonl
from src.graph_query import GraphRAGEngine
from src.graph_store import load_graph
from src.llm_client import OpenAIClient
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query GraphRAG engine.")
    parser.add_argument("--question", required=True, help="Question to answer.")
    parser.add_argument("--max-hops", type=int, default=2, help="Maximum graph hops.")
    parser.add_argument("--no-llm", action="store_true", help="Use evidence-only fallback.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    graph = load_graph(CONFIG.graph_path)
    chunks = load_jsonl(CONFIG.chunks_path)
    client = None if args.no_llm else OpenAIClient(CONFIG.openai_api_key)
    engine = GraphRAGEngine(graph, chunks, client=client, answer_model=CONFIG.answer_model)
    result = engine.answer(args.question, max_hops=args.max_hops)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

