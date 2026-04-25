"""
ingest.py
Load and parse financial documents from PDFs, text files, or CSVs.
Outputs a list of Document dicts: {id, text, source, metadata}.
"""

import os
import argparse
import json
import csv
from pathlib import Path
from tqdm import tqdm


def load_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(filepath: str) -> str:
    from pdfminer.high_level import extract_text
    return extract_text(filepath)


def load_csv(filepath: str) -> list[dict]:
    """Each row becomes its own document using Title + Content."""
    docs = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            title = row.get("Title", "").strip()
            content = row.get("Content", "").strip()
            tag = row.get("Tag", "").strip()
            if not content:
                continue
            docs.append({
                "id": f"{Path(filepath).stem}_{i}",
                "source": filepath,
                "text": f"{title}\n\n{content}".strip(),
                "metadata": {
                    "filename": Path(filepath).name,
                    "type": "csv",
                    "title": title,
                    "tag": tag
                }
            })
    return docs


def load_document(filepath: str) -> list[dict]:
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == ".csv":
        return load_csv(filepath)
    elif ext == ".pdf":
        text = load_pdf(filepath)
    elif ext in (".txt", ".md"):
        text = load_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return [{
        "id": path.stem,
        "source": str(path),
        "text": text.strip(),
        "metadata": {"filename": path.name, "type": ext.lstrip(".")}
    }]


def ingest_directory(input_dir: str) -> list[dict]:
    docs = []
    supported = {".pdf", ".txt", ".md", ".csv"}
    files = [f for f in Path(input_dir).rglob("*") if f.suffix.lower() in supported]

    for filepath in tqdm(files, desc="Loading documents"):
        try:
            loaded = load_document(str(filepath))
            docs.extend(loaded)
            print(f"  Loaded: {filepath.name} ({len(loaded)} documents)")
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
