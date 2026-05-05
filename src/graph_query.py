from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx

from src.utils import timer

STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "company",
    "corporation",
    "did",
    "for",
    "in",
    "inc",
    "inc.",
    "is",
    "of",
    "or",
    "the",
    "to",
    "who",
    "which",
}


def _question_predicates(question: str) -> list[str]:
    lower = question.lower()
    predicates: list[str] = []
    if any(word in lower for word in ("founder", "founded", "co-founded", "cofounder")):
        if lower.startswith("when") or "when was" in lower:
            predicates.append("FOUNDED_IN")
        else:
            predicates.append("FOUNDED_BY")
            predicates.append("CREATED_BY")
    if "headquarter" in lower or "where is" in lower:
        predicates.append("HEADQUARTERED_IN")
    if "acquired" in lower or "acquire" in lower:
        predicates.append("ACQUIRED_BY")
        predicates.append("ACQUIRED")
        predicates.append("OWNS")
    if "owner" in lower or "owns" in lower or "parent" in lower:
        predicates.append("OWNS")
        predicates.append("PARENT_COMPANY")
        predicates.append("SUBSIDIARY_OF")
        predicates.append("ACQUIRED_BY")
    if "invested" in lower or "investment" in lower:
        predicates.append("INVESTED_IN")
    if "industry" in lower:
        predicates.append("INDUSTRY")
    return list(dict.fromkeys(predicates))


class GraphRAGEngine:
    def __init__(
        self,
        graph: nx.MultiDiGraph,
        chunks: list[dict[str, Any]],
        client: Any | None = None,
        answer_model: str = "gpt-4o-mini",
    ) -> None:
        self.graph = graph
        self.chunks = chunks
        self.client = client
        self.answer_model = answer_model

    def match_entities(self, question: str) -> list[str]:
        question_lower = question.lower()
        nodes = list(self.graph.nodes)
        question_tokens = {
            token.strip(".,?!:;()[]'\"").lower()
            for token in question_lower.split()
        }
        question_tokens = {token for token in question_tokens if token and token not in STOPWORDS}

        direct = []
        for node in nodes:
            node_lower = node.lower()
            node_tokens = {
                token.strip(".,?!:;()[]'\"").lower()
                for token in node_lower.split()
            }
            if node_lower in question_lower or any(token in node_tokens or token in node_lower for token in question_tokens):
                direct.append(node)
        if direct:
            direct.sort(key=lambda node: (len(set(node.lower().split()) & question_tokens), self.graph.degree(node)), reverse=True)
            return direct[:5]

        scored: list[tuple[int, str]] = []
        for node in nodes:
            node_tokens = {
                token.strip(".,?!:;()[]'\"").lower()
                for token in node.lower().split()
            }
            node_tokens = {token for token in node_tokens if token and token not in STOPWORDS}
            score = len(question_tokens & node_tokens)
            if score:
                scored.append((score, node))
        scored.sort(key=lambda item: (item[0], self.graph.degree(item[1])), reverse=True)
        return [node for _, node in scored[:5]]

    def collect_neighborhood(self, entity: str, max_hops: int = 2) -> list[dict[str, Any]]:
        if entity not in self.graph:
            return []
        seen_nodes = {entity}
        seen_edges: set[tuple[str, str, int]] = set()
        queue: deque[tuple[str, int]] = deque([(entity, 0)])
        triples: list[dict[str, Any]] = []

        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
            for neighbor in neighbors:
                if neighbor not in seen_nodes:
                    seen_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))
                for source, target in ((node, neighbor), (neighbor, node)):
                    if not self.graph.has_edge(source, target):
                        continue
                    for key, data in self.graph[source][target].items():
                        edge_key = (source, target, key)
                        if edge_key in seen_edges:
                            continue
                        seen_edges.add(edge_key)
                        triples.append(
                            {
                                "subject": source,
                                "predicate": data.get("predicate", ""),
                                "object": target,
                                "confidence": data.get("confidence", 0.0),
                                "source_chunk_ids": data.get("source_chunk_ids", []),
                                "method": data.get("method", "unknown"),
                            }
                        )
        return triples

    def textualize_triples(self, triples: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"- {triple['subject']} {triple['predicate']} {triple['object']}."
            for triple in triples
        )

    def rank_evidence(self, question: str, triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        desired = _question_predicates(question)
        question_tokens = {
            token.strip(".,?!:;()[]'\"").lower()
            for token in question.lower().split()
        }
        question_tokens = {token for token in question_tokens if token and token not in STOPWORDS}

        def score(triple: dict[str, Any]) -> tuple[int, float]:
            predicate = str(triple.get("predicate", ""))
            text = f"{triple.get('subject', '')} {predicate} {triple.get('object', '')}".lower()
            token_score = sum(1 for token in question_tokens if token in text)
            predicate_score = 8 if predicate in desired else 0
            confidence = float(triple.get("confidence", 0.0))
            return predicate_score + token_score, confidence

        return sorted(triples, key=score, reverse=True)

    def answer(self, question: str, max_hops: int = 2) -> dict[str, Any]:
        with timer() as elapsed:
            matched = self.match_entities(question)
            evidence: list[dict[str, Any]] = []
            seen = set()
            for entity in matched:
                for triple in self.collect_neighborhood(entity, max_hops=max_hops):
                    key = (triple["subject"], triple["predicate"], triple["object"])
                    if key not in seen:
                        seen.add(key)
                        evidence.append(triple)

            evidence = self.rank_evidence(question, evidence)[:35]
            evidence_text = self.textualize_triples(evidence)
            token_usage: dict[str, int] = {}
            if self.client and self.client.is_available() and evidence_text:
                prompt = (
                    "Bạn trả lời câu hỏi chỉ dựa trên evidence triples bên dưới.\n"
                    'Nếu evidence không đủ, nói rõ "Không đủ dữ kiện trong graph".\n'
                    "Nếu câu hỏi hỏi founder/co-founder, hãy liệt kê tất cả triple FOUNDED_BY/CREATED_BY liên quan.\n"
                    "Nếu câu hỏi hỏi owner/parent/acquisition, hãy theo đường nhiều hop rồi nêu cả entity trung gian nếu có.\n"
                    f"Question: {question}\n"
                    f"Evidence:\n{evidence_text}\n"
                    "Answer bằng tiếng Việt, ngắn gọn, có giải thích evidence."
                )
                try:
                    answer, token_usage = self.client.generate_text(prompt, self.answer_model)
                except Exception as exc:  # pragma: no cover - network boundary
                    answer = f"LLM lỗi, fallback evidence:\n{evidence_text}\nError: {exc}"
            else:
                answer = evidence_text or "Không đủ dữ kiện trong graph."

        return {
            "question": question,
            "answer": answer,
            "matched_entities": matched,
            "evidence_triples": evidence,
            "latency_seconds": elapsed["latency_seconds"],
            "token_usage": token_usage,
        }
