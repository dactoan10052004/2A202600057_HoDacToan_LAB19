from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src.config import CONFIG
from src.corpus_loader import load_jsonl
from src.evaluator import run_benchmark
from src.flat_rag import FlatRAGEngine
from src.graph_query import GraphRAGEngine
from src.graph_store import load_graph
from src.llm_client import OpenAIClient
from src.utils import read_json, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Flat RAG vs GraphRAG benchmark.")
    parser.add_argument("--no-llm", action="store_true", help="Force local fallbacks.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    questions = read_json(CONFIG.benchmark_questions_path)
    chunks = load_jsonl(CONFIG.chunks_path)
    graph = load_graph(CONFIG.graph_path)
    client = None if args.no_llm else OpenAIClient(CONFIG.openai_api_key)

    flat_engine = FlatRAGEngine(client=client, answer_model=CONFIG.answer_model, embedding_model=CONFIG.embedding_model)
    flat_engine.build_index(chunks)
    graph_engine = GraphRAGEngine(graph, chunks, client=client, answer_model=CONFIG.answer_model)

    df = run_benchmark(questions, flat_engine, graph_engine)
    CONFIG.benchmark_report_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CONFIG.benchmark_report_path, index=False, encoding="utf-8-sig")

    flat_accuracy = float(df["flat_rag_correct"].mean()) if len(df) else 0.0
    graph_accuracy = float(df["graphrag_correct"].mean()) if len(df) else 0.0
    graph_only = int((~df["flat_rag_correct"] & df["graphrag_correct"]).sum()) if len(df) else 0
    print(f"Saved report: {CONFIG.benchmark_report_path}")
    print(f"Flat accuracy: {flat_accuracy:.2%}")
    print(f"Graph accuracy: {graph_accuracy:.2%}")
    print(f"Flat avg latency: {df['flat_rag_latency'].mean():.3f}s")
    print(f"Graph avg latency: {df['graphrag_latency'].mean():.3f}s")
    print(f"Flat avg tokens: {df['flat_rag_tokens'].mean():.1f}")
    print(f"Graph avg tokens: {df['graphrag_tokens'].mean():.1f}")
    print(f"GraphRAG correct but Flat RAG wrong: {graph_only}")


if __name__ == "__main__":
    main()
