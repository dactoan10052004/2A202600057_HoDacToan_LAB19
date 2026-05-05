from src.chunker import chunk_text


def test_chunk_text_returns_non_empty_chunks() -> None:
    chunks = chunk_text("OpenAI was founded in 2015. " * 50, chunk_size=120, overlap=20)
    assert chunks
    assert chunks[0]["chunk_id"] == "chunk_000001"
    assert all(chunk["text"] for chunk in chunks)


def test_chunk_text_overlap_works() -> None:
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    assert len(chunks) > 1
    assert int(chunks[1]["start_char"]) < int(chunks[0]["end_char"])

