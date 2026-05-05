# Lab Day 19 Report

## 1. Mapping theo yêu cầu bài lab

- Entity Extraction / Relation Extraction: `src/extractor.py` dùng OpenAI Responses API để trích xuất triples, đồng thời merge structured facts từ Wikipedia infobox.
- Graph Construction: `src/graph_builder.py` dựng `networkx.MultiDiGraph`; `src/normalizer.py` xử lý alias và deduplicate.
- Query Answering: `src/graph_query.py` match entity, duyệt graph 2-hop theo cả successors/predecessors, textualize evidence và gửi LLM trả lời.
- Flat RAG baseline: `src/flat_rag.py` dùng OpenAI embeddings + cosine similarity, fallback TF-IDF khi không có key.
- Evaluation: `scripts/06_run_benchmark.py` chạy 20 câu hỏi và lưu CSV.

## 2. Kết quả graph mới nhất

- Graph nodes: 767
- Graph edges: 987
- Chunks indexed: 64
- Extraction mode: LLM + structured fallback
- Graph image: `outputs/graph.png`
- Graph stats: `outputs/graph_stats.json`

## 3. Benchmark 20 câu hỏi

File chi tiết: `outputs/benchmark_report.csv`

- Flat RAG accuracy: 95.00%
- GraphRAG accuracy: 100.00%
- Flat avg latency: 1.876s
- Graph avg latency: 2.760s
- Flat avg tokens: 1458.3
- Graph avg tokens: 586.2
- Cases GraphRAG correct but Flat RAG wrong: 1

Case nổi bật:

- `q018`: "Who founded the company that Microsoft invested in?"
- Expected: `Sam Altman`
- Lý do GraphRAG tốt hơn: graph có đường multi-hop `Microsoft -> INVESTED_IN -> OpenAI -> FOUNDED_BY -> Sam Altman`, trong khi Flat RAG dễ retrieve thiếu một trong hai mảnh evidence.

## 4. Chi phí/time khi xây graph

File chi tiết: `outputs/indexing_cost_summary.json`

- Indexing time: 565.6905 seconds
- Prompt tokens: 29,723
- Completion tokens: 27,503
- Total extraction tokens: 57,226

## 5. Ghi chú công cụ

- NetworkX: dùng chính trong project vì chạy offline tốt, dễ inspect thuật toán BFS/multi-hop.
- Neo4j: phù hợp nếu cần giao diện trực quan và Cypher cho graph lớn hơn.
- NodeRAG: phù hợp nếu muốn framework GraphRAG trọn gói, nhưng lab này tự implement để thể hiện rõ pipeline.
