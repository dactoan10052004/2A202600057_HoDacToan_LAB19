from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from src.utils import safe_json_loads


class OpenAIClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.last_usage: dict[str, int] = {}

    def is_available(self) -> bool:
        return self.client is not None

    def _require_client(self) -> OpenAI:
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or use fallback mode.")
        return self.client

    @staticmethod
    def _usage_to_dict(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            raw = usage.model_dump()
        elif isinstance(usage, dict):
            raw = usage
        else:
            raw = usage.__dict__
        return {
            "prompt_tokens": int(raw.get("input_tokens") or raw.get("prompt_tokens") or 0),
            "completion_tokens": int(raw.get("output_tokens") or raw.get("completion_tokens") or 0),
            "total_tokens": int(raw.get("total_tokens") or 0),
        }

    @staticmethod
    def _output_text(response: Any) -> str:
        direct = getattr(response, "output_text", None)
        if direct:
            return str(direct)
        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()

    def generate_json(self, prompt: str, model: str) -> dict[str, Any]:
        text, _ = self.generate_text(prompt, model)
        try:
            parsed = safe_json_loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON: {text[:500]}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Model JSON response must be an object")
        return parsed

    def generate_text(self, prompt: str, model: str) -> tuple[str, dict[str, int]]:
        client = self._require_client()
        response = client.responses.create(model=model, input=prompt)
        usage = self._usage_to_dict(response)
        self.last_usage = usage
        return self._output_text(response), usage

    def embed_texts(self, texts: list[str], model: str) -> list[list[float]]:
        client = self._require_client()
        response = client.embeddings.create(model=model, input=texts)
        self.last_usage = self._usage_to_dict(response)
        return [list(item.embedding) for item in response.data]
