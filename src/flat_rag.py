from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import timer


class FlatRAGEngine:
    def __init__(
        self,
        client: Any | None = None,
        answer_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.client = client
        self.answer_model = answer_model
        self.embedding_model = embedding_model
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: Any | None = None

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        texts = [str(chunk.get("text", "")) for chunk in chunks]
        if not texts:
            return
        if self.client and self.client.is_available():
            self.embeddings = np.array(self.client.embed_texts(texts, self.embedding_model), dtype=float)
            return
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.chunks:
            return []
        if self.embeddings is not None and self.client and self.client.is_available():
            query_vector = np.array(self.client.embed_texts([question], self.embedding_model), dtype=float)
            scores = cosine_similarity(query_vector, self.embeddings)[0]
        else:
            if self.vectorizer is None or self.tfidf_matrix is None:
                raise RuntimeError("Index is not built")
            query_vector = self.vectorizer.transform([question])
            scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for index in indices:
            chunk = dict(self.chunks[int(index)])
            chunk["score"] = float(scores[int(index)])
            results.append(chunk)
        return results

    def answer(self, question: str, top_k: int = 5) -> dict[str, Any]:
        with timer() as elapsed:
            retrieved = self.retrieve(question, top_k=top_k)
            context = "\n\n".join(f"[{row.get('chunk_id')}]\n{row.get('text')}" for row in retrieved)
            token_usage: dict[str, int] = {}
            if self.client and self.client.is_available() and context:
                prompt = (
                    "Trả lời câu hỏi bằng tiếng Việt, chỉ dựa vào context dưới đây. "
                    "Nếu thiếu dữ kiện, nói rõ không đủ dữ kiện.\n"
                    f"Question: {question}\nContext:\n{context}"
                )
                try:
                    answer, token_usage = self.client.generate_text(prompt, self.answer_model)
                except Exception as exc:  # pragma: no cover - network boundary
                    answer = f"LLM lỗi, fallback top chunks:\n{context}\nError: {exc}"
            else:
                answer = context or "Không có chunk phù hợp."
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "latency_seconds": elapsed["latency_seconds"],
            "token_usage": token_usage,
        }
