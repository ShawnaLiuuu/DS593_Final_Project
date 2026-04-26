"""
pipeline.py
End-to-end RAG pipeline: load config → chunk → index → retrieve → generate.

Usage:
    python src/pipeline.py --query "Why did revenue decrease?" --config configs/config.yaml
"""

import json
import argparse
import yaml
from pathlib import Path

from ingest import ingest_directory
from chunker import chunk_all
from embedder import Embedder
from retriever import Retriever
from generator import Generator
from dotenv import load_dotenv
load_dotenv()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_pipeline(config: dict):
    # 1. Ingest
    raw_dir = config["paths"]["raw_data"]
    processed_dir = config["paths"]["processed"]
    doc_cache = Path(processed_dir) / "documents.json"

    if doc_cache.exists():
        print(f"Loading cached documents from {doc_cache}")
        with open(doc_cache) as f:
            documents = json.load(f)
    else:
        documents = ingest_directory(raw_dir)
        Path(processed_dir).mkdir(parents=True, exist_ok=True)
        with open(doc_cache, "w") as f:
            json.dump(documents, f, indent=2)

    # 2. Chunk
    chunks = chunk_all(
        documents,
        strategy=config["chunking"]["strategy"],
        chunk_size=config["chunking"]["chunk_size"],
        overlap=config["chunking"]["overlap"]
    )

    # 3. Build retriever
    method = config["retrieval"]["method"]
    embedder = None
    if method in ("semantic", "hybrid"):
        embedder = Embedder(config["embedding"]["model"])

    retriever = Retriever(
        chunks=chunks,
        method=method,
        embedder=embedder,
        hybrid_alpha=config["retrieval"]["hybrid_alpha"]
    )

    # 4. Build generator
    generator = Generator(
        model=config["generation"]["model"],
        temperature=config["generation"]["temperature"],
        max_tokens=config["generation"]["max_tokens"],
        prompt_strategy=config["generation"]["prompt_strategy"]
    )

    return retriever, generator, config["retrieval"]["top_k"]


def run_query(query: str, retriever: Retriever, generator: Generator,
              top_k: int, baseline: bool = False) -> dict:
    if baseline:
        answer = generator.generate(query, retrieved=None)
        return {"query": query, "mode": "baseline", "answer": answer, "retrieved": []}

    retrieved = retriever.retrieve(query, top_k=top_k)
    answer = generator.generate(query, retrieved=retrieved)
    return {
        "query": query,
        "mode": retriever.method,
        "answer": answer,
        "retrieved": [
            {"chunk_id": c.id, "doc_id": c.doc_id, "score": s, "text_preview": c.text[:200]}
            for c, s in retrieved
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--baseline", action="store_true",
                        help="Run without retrieval (baseline)")
    args = parser.parse_args()

    config = load_config(args.config)
    retriever, generator, top_k = build_pipeline(config)

    result = run_query(args.query, retriever, generator, top_k, baseline=args.baseline)

    print(f"\n{'='*60}")
    print(f"Query: {result['query']}")
    print(f"Mode:  {result['mode']}")
    print(f"\nAnswer:\n{result['answer']}")
    if result["retrieved"]:
        print(f"\nTop {len(result['retrieved'])} retrieved chunks:")
        for i, r in enumerate(result["retrieved"], 1):
            print(f"  [{i}] {r['doc_id']} (score: {r['score']:.3f}): {r['text_preview']}...")
