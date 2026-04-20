"""
ingest.py
Load and parse financial documents (PDFs, text files) from a directory.
Outputs a list of Document dicts: {id, text, source, metadata}.
"""

import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm


def load_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(filepath: str) -> str:
    from pdfminer.high_level import extract_text
    return extract_text(filepath)


def load_document(filepath: str) -> dict:
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == ".pdf":
        text = load_pdf(filepath)
    elif ext in (".txt", ".md"):
        text = load_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return {
        "id": path.stem,
        "source": str(path),
        "text": text.strip(),
        "metadata": {
            "filename": path.name,
            "type": ext.lstrip(".")
        }
    }


def ingest_directory(input_dir: str) -> list[dict]:
    docs = []
    supported = {".pdf", ".txt", ".md"}
    files = [f for f in Path(input_dir).rglob("*") if f.suffix.lower() in supported]

    for filepath in tqdm(files, desc="Loading documents"):
        try:
            doc = load_document(str(filepath))
            docs.append(doc)
            print(f"  Loaded: {filepath.name} ({len(doc['text'])} chars)")
        except Exception as e:
            print(f"  [WARN] Failed to load {filepath.name}: {e}")

    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with raw documents")
    parser.add_argument("--output", default="data/processed/documents.json")
    args = parser.parse_args()

    documents = ingest_directory(args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(documents, f, indent=2)

    print(f"\nIngested {len(documents)} documents → {args.output}")
