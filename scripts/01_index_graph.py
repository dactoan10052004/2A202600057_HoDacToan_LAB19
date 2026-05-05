from __future__ import annotations

import argparse
import json
import time

from tqdm import tqdm

import _bootstrap  # noqa: F401
from src.chunker import chunk_text
from src.config import CONFIG
from src.corpus_loader import ensure_dirs, load_text_file, save_jsonl
from src.extractor import extract_triples_from_chunk
from src.graph_builder import build_graph, compute_graph_stats
from src.graph_store import save_graph, save_graph_stats
from src.llm_client import OpenAIClient
from src.normalizer import deduplicate_triples
from src.cost_tracker import sum_token_usage
from src.utils import setup_logging, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk corpus, extract triples, and build graph index.")
    parser.add_argument("--input", default=str(CONFIG.raw_txt_path), help="Input text corpus path.")
    parser.add_argument("--chunk-size", type=int, default=900, help="Characters per chunk.")
    parser.add_argument("--overlap", type=int, default=150, help="Character overlap between chunks.")
    parser.add_argument("--no-llm", action="store_true", help="Force rule-based extraction fallback.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    start_time = time.perf_counter()
    args = parse_args()
    ensure_dirs()
    text = load_text_file(args.input)
    chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap, source=str(args.input))
    save_jsonl(chunks, CONFIG.chunks_path)

    client = None if args.no_llm else OpenAIClient(CONFIG.openai_api_key)
    triples = []
    extraction_token_usages = []
    for chunk in tqdm(chunks, desc="Extracting triples"):
        before_usage = dict(getattr(client, "last_usage", {})) if client else {}
        triples.extend(extract_triples_from_chunk(chunk, client))
        after_usage = dict(getattr(client, "last_usage", {})) if client else {}
        if after_usage and after_usage != before_usage:
            extraction_token_usages.append({"token_usage": after_usage})
    triples = deduplicate_triples(triples)
    save_jsonl(triples, CONFIG.triples_path)

    graph = build_graph(triples)
    stats = compute_graph_stats(graph)
    save_graph(graph, CONFIG.graph_path)
    save_graph_stats(stats, CONFIG.graph_stats_path)
    elapsed = round(time.perf_counter() - start_time, 4)
    cost_summary = {
        "mode": "rule_based" if args.no_llm or client is None or not client.is_available() else "llm",
        "chunks": len(chunks),
        "triples": len(triples),
        "graph_nodes": stats["num_nodes"],
        "graph_edges": stats["num_edges"],
        "indexing_seconds": elapsed,
        "extraction_token_usage": sum_token_usage(extraction_token_usages),
        "notes": "Token usage is captured for OpenAI extraction calls; rule-based fallback uses zero tokens.",
    }
    write_json(cost_summary, CONFIG.indexing_cost_summary_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Chunks: {CONFIG.chunks_path}")
    print(f"Triples: {CONFIG.triples_path}")
    print(f"Graph: {CONFIG.graph_path}")
    print(f"Indexing cost summary: {CONFIG.indexing_cost_summary_path}")


if __name__ == "__main__":
    main()
