"""Document loading and chunking for the ingestion pipeline."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    title: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _split_into_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def recursive_chunk(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Recursively split text on paragraph -> sentence -> char boundaries."""
    paragraphs = _split_into_paragraphs(text)
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) <= chunk_size:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        if len(para) <= chunk_size:
            buf = para
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sbuf = ""
            for sent in sentences:
                scandidate = f"{sbuf} {sent}".strip()
                if len(scandidate) <= chunk_size:
                    sbuf = scandidate
                else:
                    if sbuf:
                        chunks.append(sbuf)
                    sbuf = sent[:chunk_size]
            buf = sbuf
    if buf:
        chunks.append(buf)

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = overlapped[-1][-overlap:]
            overlapped.append(f"{tail} {chunks[i]}".strip())
        return overlapped
    return chunks


def load_and_chunk_file(path: Path, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    title = path.stem.replace("_", " ").title()
    pieces = recursive_chunk(text, chunk_size=chunk_size, overlap=overlap)
    return [
        Chunk(
            id=str(uuid.uuid4()),
            text=piece,
            source=str(path),
            title=title,
            chunk_index=i,
            metadata={"filename": path.name},
        )
        for i, piece in enumerate(pieces)
    ]


def load_and_chunk_directory(directory: Path, pattern: str = "*.txt", **kwargs) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.glob(pattern)):
        chunks.extend(load_and_chunk_file(path, **kwargs))
    return chunks
