from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None
    extraction_model: str
    answer_model: str
    embedding_model: str
    wiki_user_agent: str
    raw_jsonl_path: Path
    raw_txt_path: Path
    chunks_path: Path
    triples_path: Path
    graph_path: Path
    graph_png_path: Path
    graph_stats_path: Path
    indexing_cost_summary_path: Path
    benchmark_questions_path: Path
    benchmark_report_path: Path

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv(PROJECT_ROOT / ".env")
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            extraction_model=os.getenv("EXTRACTION_MODEL", "gpt-4o-mini"),
            answer_model=os.getenv("ANSWER_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            wiki_user_agent=os.getenv(
                "WIKI_USER_AGENT",
                "GraphRAGLab/1.0 (your_email@example.com)",
            ),
            raw_jsonl_path=PROJECT_ROOT / "data" / "raw" / "tech_company_corpus.jsonl",
            raw_txt_path=PROJECT_ROOT / "data" / "raw" / "tech_company_corpus.txt",
            chunks_path=PROJECT_ROOT / "data" / "processed" / "chunks.jsonl",
            triples_path=PROJECT_ROOT / "data" / "processed" / "triples.jsonl",
            graph_path=PROJECT_ROOT / "data" / "processed" / "graph.pkl",
            graph_png_path=PROJECT_ROOT / "outputs" / "graph.png",
            graph_stats_path=PROJECT_ROOT / "outputs" / "graph_stats.json",
            indexing_cost_summary_path=PROJECT_ROOT / "outputs" / "indexing_cost_summary.json",
            benchmark_questions_path=PROJECT_ROOT / "data" / "benchmark" / "questions.json",
            benchmark_report_path=PROJECT_ROOT / "outputs" / "benchmark_report.csv",
        )

    def ensure_output_dirs(self) -> None:
        for path in (
            self.raw_jsonl_path,
            self.raw_txt_path,
            self.chunks_path,
            self.triples_path,
            self.graph_path,
            self.graph_png_path,
            self.graph_stats_path,
            self.indexing_cost_summary_path,
            self.benchmark_questions_path,
            self.benchmark_report_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)


CONFIG = AppConfig.from_env()
