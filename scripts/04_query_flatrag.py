from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from src.config import CONFIG
from src.corpus_loader import load_jsonl
from src.flat_rag import FlatRAGEngine
from src.llm_client import OpenAIClient
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Flat RAG baseline.")
    parser.add_argument("--question", required=True, help="Question to answer.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--no-llm", action="store_true", help="Use TF-IDF and top-chunk fallback.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    chunks = load_jsonl(CONFIG.chunks_path)
    client = None if args.no_llm else OpenAIClient(CONFIG.openai_api_key)
    engine = FlatRAGEngine(client=client, answer_model=CONFIG.answer_model, embedding_model=CONFIG.embedding_model)
    engine.build_index(chunks)
    result = engine.answer(args.question, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

