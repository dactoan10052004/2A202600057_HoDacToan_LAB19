from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import CONFIG


def ensure_dirs() -> None:
    CONFIG.ensure_output_dirs()


def load_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

