# Lab Day 19: GraphRAG với Tech Company Corpus

Project này xây dựng pipeline GraphRAG end-to-end bằng Python 3.10+: cào dữ liệu Wikipedia về các công ty công nghệ, chunk corpus, trích xuất triples, xây graph NetworkX, query multi-hop, so sánh với Flat RAG baseline và xuất benchmark.

## Kiến trúc

- `src/data_scraper.py`: lấy Wikipedia REST summaries với `User-Agent` rõ ràng và rate limit nhẹ.
  Phiên bản hiện tại còn lấy thêm Wikipedia API full extract và một số trường infobox quan trọng như founders, founded, headquarters, owner, parent, industry, products.
- `src/chunker.py`: chia corpus thành chunk có overlap.
- `src/extractor.py`: trích xuất triples bằng OpenAI Responses API, sau đó merge thêm structured rule-based triples từ infobox để tăng độ phủ; fallback regex khi thiếu key hoặc API lỗi.
- `src/graph_builder.py`: dựng `networkx.MultiDiGraph`.
- `src/graph_store.py`: lưu/load graph bằng `pickle.dump` và `pickle.load`, không dùng `write_gpickle`.
- `src/graph_query.py`: GraphRAG multi-hop trên predecessors/successors.
- `src/flat_rag.py`: baseline cosine similarity bằng OpenAI embeddings hoặc TF-IDF fallback.
- `src/evaluator.py`: chạy benchmark 20 câu hỏi và xuất CSV.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Trên Windows PowerShell, nếu không có `cp`:

```powershell
Copy-Item .env.example .env
```

Điền `OPENAI_API_KEY` trong `.env` nếu muốn dùng LLM. Nếu để trống, project vẫn chạy chế độ demo offline bằng rule-based extraction và TF-IDF.

## Chạy từ đầu

```bash
python scripts/00_scrape_data.py --limit 30
python scripts/01_index_graph.py --chunk-size 1500 --overlap 200
python scripts/02_visualize_graph.py
python scripts/03_query_graphrag.py --question "Who founded OpenAI?"
python scripts/04_query_flatrag.py --question "Who founded OpenAI?"
python scripts/05_create_benchmark.py
python scripts/06_run_benchmark.py
pytest
```

Ép chạy không dùng LLM:

```bash
python scripts/01_index_graph.py --no-llm
python scripts/03_query_graphrag.py --question "Who founded OpenAI?" --no-llm
python scripts/04_query_flatrag.py --question "Who founded OpenAI?" --no-llm
python scripts/06_run_benchmark.py --no-llm
```

## Output files

- `data/raw/tech_company_corpus.jsonl`: dữ liệu Wikipedia dạng JSONL.
- `data/raw/tech_company_corpus.txt`: corpus text gộp.
- `data/processed/chunks.jsonl`: chunks.
- `data/processed/triples.jsonl`: triples đã normalize/dedup.
- `data/processed/graph.pkl`: graph NetworkX lưu bằng pickle.
- `outputs/graph.png`: ảnh graph.
- `outputs/graph_stats.json`: thống kê graph.
- `outputs/indexing_cost_summary.json`: thời gian và token usage khi xây graph.
- `data/benchmark/questions.json`: 20 câu hỏi benchmark.
- `outputs/benchmark_report.csv`: kết quả Flat RAG vs GraphRAG.
- `outputs/lab_report.md`: tóm tắt theo deliverables của lab.

## Bám sát yêu cầu lab

- Entity/Relation Extraction: LLM đọc text và trả triples; rule-based layer chỉ bổ sung structured facts từ Wikipedia infobox để giảm thiếu dữ kiện.
- Graph Construction: dùng `NetworkX MultiDiGraph`, normalize alias và deduplicate triples trước khi build graph.
- Multi-hop Querying: `GraphRAGEngine` match entity, duyệt predecessors/successors trong phạm vi 2-hop, textualize triples rồi gửi evidence cho LLM.
- Flat RAG baseline: dùng OpenAI embeddings + cosine similarity; không cần FAISS/ChromaDB cho lab nhỏ.
- Evaluation: 20 câu hỏi benchmark, có latency, token usage và số evidence triples.
- Visualization: dùng Matplotlib + NetworkX, lưu `outputs/graph.png`.
- Tool recommendations: README và report nêu rõ NetworkX là lựa chọn offline chính; Neo4j/NodeRAG là hướng mở rộng phù hợp theo đề.

## Troubleshooting

- Thiếu `OPENAI_API_KEY`: hệ thống tự fallback, nhưng chất lượng triples và câu trả lời sẽ thấp hơn LLM.
- Wikipedia request fail: kiểm tra mạng và `WIKI_USER_AGENT`; scraper bỏ qua page lỗi thay vì dừng toàn bộ.
- Graph quá rối khi visualize: dùng `--max-edges`, ví dụ `python scripts/02_visualize_graph.py --max-edges 40`.
- NetworkX 3.x bỏ gpickle helpers: project dùng `pickle.dump` và `pickle.load`, nên không gặp lỗi `write_gpickle/read_gpickle`.
