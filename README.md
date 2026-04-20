# DS593 Final Project: RAG for Financial Document Q&A

**Team:** HsiangEn (Shawna) Liu, Moses Chen  
**Course:** DS593 — Boston University

## Overview

A Retrieval-Augmented Generation (RAG) pipeline for question answering over financial documents (SEC 10-K filings and earnings call transcripts). The system lets users ask natural language questions and receive grounded, document-based answers.

The project systematically compares retrieval strategies (BM25 vs. semantic embedding), chunking approaches (size, overlap), and prompting methods to evaluate their effect on answer quality.

---

## Project Structure

```
DS593_Final_Project/
├── data/
│   ├── raw/                  # Downloaded SEC filings / transcripts (gitignored)
│   └── processed/            # Chunked documents, embeddings cache
├── src/
│   ├── ingest.py             # Load and parse documents
│   ├── chunker.py            # Chunking strategies (fixed, overlap, sentence)
│   ├── retriever.py          # BM25 + semantic retrieval
│   ├── embedder.py           # Embedding model wrapper
│   ├── generator.py          # LLM generation + prompting strategies
│   └── pipeline.py           # End-to-end RAG pipeline
├── eval/
│   ├── golden_set.json       # 15–20 hand-labeled Q&A pairs
│   └── evaluate.py           # Retrieval accuracy + answer scoring
├── notebooks/
│   └── experiments.ipynb     # Exploratory analysis and ablations
├── configs/
│   └── config.yaml           # Hyperparameters: chunk size, overlap, top-k, etc.
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Approach

### Retrieval
- **BM25** (keyword-based) via `rank_bm25`
- **Semantic retrieval** via cosine similarity over sentence-transformer embeddings
- **Hybrid** (optional): combine BM25 + semantic scores

### Chunking Strategies
| Strategy | Description |
|---|---|
| Fixed-size | Split by token count, no overlap |
| Overlapping | Fixed-size with configurable overlap window |
| Sentence-aware | Split on sentence boundaries |

### Generation
- LLM: OpenAI ChatGPT via API (default), swappable
- Prompting: base prompting vs. structured (chain-of-thought / citation-grounded)

### Baseline
LLM with no retrieval (pure parametric knowledge), evaluated on the same golden set.

---

## Data Sources

- **SEC 10-K Filings:** [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar)
- **Earnings Call Transcripts / Financial News:** [Kaggle – Financial News Dataset](https://www.kaggle.com/datasets/yogeshchary/financial-news-dataset)

> Raw data files are gitignored. See `data/README.md` for download instructions.

---

## Evaluation

Golden set: ~15–20 Q&A pairs manually labeled from source documents.

| Metric | Description |
|---|---|
| Retrieval Accuracy | % of queries where relevant chunk is in top-k results |
| Answer Correctness | Manual or rubric-based scoring (0/1 or 0–3 scale) |

Ablations: chunk size × retrieval method × prompt strategy.

---

## Setup

```bash
git clone https://github.com/ShawnaLiuuu/DS593_Final_Project.git
cd DS593_Final_Project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # Add your OpenAI API key
```

---

## Usage

```bash
# Ingest and chunk documents
python src/ingest.py --input data/raw/ --output data/processed/

# Run the full pipeline on a question
python src/pipeline.py --query "Why did revenue decrease this quarter?" --config configs/config.yaml

# Run evaluation
python eval/evaluate.py --golden eval/golden_set.json
```

---

## Experiments Tracker

| Config | Chunk Size | Overlap | Retrieval | Retrieval Acc | Answer Score |
|---|---|---|---|---|---|
| Baseline (no RAG) | — | — | None | — | TBD |
| BM25-small | 256 | 0 | BM25 | TBD | TBD |
| Semantic-small | 256 | 0 | Semantic | TBD | TBD |
| Semantic-overlap | 512 | 64 | Semantic | TBD | TBD |
| Hybrid | 512 | 64 | BM25+Semantic | TBD | TBD |

---

## Ethical Considerations

Financial documents are sensitive; incorrect outputs could affect real decisions. We will document hallucination rates, incomplete answers, and biases in document interpretation, and discuss deployment limitations.
