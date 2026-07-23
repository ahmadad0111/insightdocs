from src.rag.ingestion.chunker import TextChunker, Chunker


def test_chunker_preserves_page_numbers():
    pages = [
        {"page_number": 1, "text": "word " * 300},
        {"page_number": 2, "text": "token " * 300},
    ]
    chunks = TextChunker(chunk_size=200, chunk_overlap=20).chunk_pages(pages)
    assert len(chunks) > 2
    assert {c["page_number"] for c in chunks} == {1, 2}
    assert all("text" in c and c["text"].strip() for c in chunks)
    # ids are unique and contiguous
    ids = [c["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks)))


def test_chunker_alias():
    assert Chunker is TextChunker
