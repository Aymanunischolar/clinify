from retrieval.chunking import recursive_chunk


def test_short_text_stays_one_chunk():
    text = "Short paragraph about hypertension."
    chunks = recursive_chunk(text, chunk_size=800, overlap=100)
    assert chunks == [text]


def test_long_text_splits_into_multiple_chunks():
    paragraph = "Hypertension is a major cardiovascular risk factor. " * 40
    text = f"{paragraph}\n\n{paragraph}"
    chunks = recursive_chunk(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 50 for c in chunks)


def test_chunks_are_non_empty():
    text = "A" * 5000
    chunks = recursive_chunk(text, chunk_size=500, overlap=50)
    assert all(c.strip() for c in chunks)
