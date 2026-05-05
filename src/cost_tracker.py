from __future__ import annotations

from typing import Any


def extract_token_usage(result: dict[str, Any]) -> dict[str, int]:
    usage = result.get("token_usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }


def sum_token_usage(results: list[dict[str, Any]]) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for result in results:
        usage = extract_token_usage(result)
        for key in total:
            total[key] += usage[key]
    return total

