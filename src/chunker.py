"""
chunker.py
Split documents into chunks using three strategies:
  - fixed:    fixed token count, no overlap
  - overlap:  fixed token count with sliding window overlap
  - sentence: split on sentence boundaries
"""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    chunk_index: int
    metadata: dict


def _split_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping token windows (approximated by whitespace split)."""
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def _split_sentences(text: str, chunk_size: int) -> list[str]:
    """Split text into sentence-boundary chunks up to ~chunk_size tokens."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, count = [], [], 0
    for sent in sentences:
        tokens = len(sent.split())
        if count + tokens > chunk_size and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(doc: dict, strategy: str = "overlap",
                   chunk_size: int = 512, overlap: int = 64) -> list[Chunk]:
    text = doc["text"]
    doc_id = doc["id"]

    if strategy == "fixed":
        texts = _split_tokens(text, chunk_size, overlap=0)
    elif strategy == "overlap":
        texts = _split_tokens(text, chunk_size, overlap)
    elif strategy == "sentence":
        texts = _split_sentences(text, chunk_size)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return [
        Chunk(
            id=f"{doc_id}_chunk{i}",
            doc_id=doc_id,
            text=t,
            chunk_index=i,
            metadata={**doc.get("metadata", {}), "strategy": strategy,
                      "chunk_size": chunk_size, "overlap": overlap}
        )
        for i, t in enumerate(texts) if t.strip()
    ]


def chunk_all(documents: list[dict], **kwargs) -> list[Chunk]:
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, **kwargs))
    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents.")
    return all_chunks
