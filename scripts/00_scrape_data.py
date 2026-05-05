from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src.config import CONFIG
from src.corpus_loader import ensure_dirs
from src.data_scraper import DEFAULT_COMPANY_PAGES, RELATED_ENTITY_PAGES, save_corpus, scrape_companies
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Wikipedia summaries for tech companies.")
    parser.add_argument("--limit", type=int, default=30, help="Maximum number of companies to scrape.")
    parser.add_argument(
        "--include-related",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also scrape related pages used by benchmark, such as YouTube and Facebook.",
    )
    parser.add_argument("--output-jsonl", default=str(CONFIG.raw_jsonl_path), help="JSONL output path.")
    parser.add_argument("--output-txt", default=str(CONFIG.raw_txt_path), help="Text corpus output path.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    ensure_dirs()
    titles = DEFAULT_COMPANY_PAGES[: args.limit]
    if args.include_related:
        titles = titles + [title for title in RELATED_ENTITY_PAGES if title not in titles]
    rows = scrape_companies(titles, CONFIG.wiki_user_agent)
    save_corpus(rows, args.output_jsonl, args.output_txt)
    print(f"Scraped {len(rows)} documents")
    print(f"JSONL: {args.output_jsonl}")
    print(f"TXT: {args.output_txt}")


if __name__ == "__main__":
    main()
