from __future__ import annotations

import re


def _find_word_boundary(text: str, start: int, proposed_end: int) -> int:
    if proposed_end >= len(text):
        return len(text)
    boundary = text.rfind(" ", start, proposed_end)
    if boundary <= start:
        return proposed_end
    return boundary


def _section_ranges(text: str) -> list[tuple[str | None, int, int]]:
    headers = list(re.finditer(r"(?m)^#\s+(.+)$", text))
    if not headers:
        return [(None, 0, len(text))]
    ranges: list[tuple[str | None, int, int]] = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        ranges.append((header.group(1).strip(), header.start(), end))
    return ranges


def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 150,
    source: str = "tech_company_corpus.txt",
) -> list[dict[str, object]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[dict[str, object]] = []
    for title, section_start, section_end in _section_ranges(text):
        start = section_start
        while start < section_end:
            end = min(_find_word_boundary(text, start, start + chunk_size), section_end)
            chunk = text[start:end].strip()
            if chunk:
                row: dict[str, object] = {
                    "chunk_id": f"chunk_{len(chunks) + 1:06d}",
                    "text": chunk,
                    "start_char": start,
                    "end_char": end,
                    "source": source,
                }
                if title:
                    row["source_title"] = title
                chunks.append(row)
            if end >= section_end:
                break
            next_start = max(section_start, end - overlap)
            while next_start < section_end and next_start > section_start and text[next_start - 1].isalnum():
                next_start += 1
            start = next_start
    return chunks
