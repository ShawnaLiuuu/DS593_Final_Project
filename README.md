# DS593 Final Project: RAG for Financial Document Q&A

**Team:** HsiangEn (Shawna) Liu, Moses Chen  
**Course:** DS593 — Boston University

## Overview

A Retrieval-Augmented Generation (RAG) pipeline for question answering over financial documents (SEC 10-K filings and financial news articles). The system lets users ask natural language questions and receive grounded, document-based answers via an interactive Streamlit UI.

The project systematically compares retrieval strategies (BM25, semantic, hybrid), chunking approaches (size, overlap), and prompting methods to evaluate their effect on answer quality.

---

## Project Structure

```
DS593_Final_Project/
├── app.py                    # Streamlit UI
├── data/
│   ├── raw/                  # Raw documents (gitignored)
│   │   ├── apple_10k_2025.htm
│   │   ├── Financial_Categorized.csv
│   │   └── download_data.py  # Script to download Kaggle dataset
│   └── processed/            # Embeddings cache (gitignored)
├── src/
│   ├── ingest.py             # Load and parse documents (CSV, HTML, PDF)
│   ├── chunker.py            # Chunking strategies (fixed, overlap, sentence)
│   ├── retriever.py          # BM25 + semantic + hybrid retrieval
│   ├── embedder.py           # Embedding model wrapper with disk cache
│   ├── generator.py          # LLM generation + prompting strategies
│   └── pipeline.py           # End-to-end RAG pipeline
├── eval/
│   ├── golden_set.json               # 15 hand-labeled Q&A pairs
│   ├── evaluate.py                   # Retrieval accuracy + token overlap scoring
│   ├── results_semantic_512.json     # Ablation results
│   ├── results_bm25_512.json
│   ├── results_semantic_256.json
│   └── results_hybrid_512.json
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
- **Semantic retrieval** via cosine similarity over sentence-transformer embeddings (`all-MiniLM-L6-v2`)
- **Hybrid**: weighted combination of BM25 + semantic scores

### Chunking Strategies
| Strategy | Description |
|---|---|
| Fixed-size | Split by token count, no overlap |
| Overlapping | Fixed-size with configurable overlap window |
| Sentence-aware | Split on sentence boundaries |

### Generation
- LLM: OpenAI GPT-4o Mini via API
- Prompting: base prompting vs. structured citation-grounded prompting

### Baseline
LLM with no retrieval (pure parametric knowledge), evaluated on the same golden set.

---

## Data Sources

- **Apple 10-K FY2025:** [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar) — `aapl-20250927.htm`
- **Financial News (62k articles):** [Kaggle – Financial News Dataset](https://www.kaggle.com/datasets/yogeshchary/financial-news-dataset)
- **SEC Filings Corpus:** [HuggingFace – PleIAs/SEC](https://huggingface.co/datasets/PleIAs/SEC)

> Raw data files are gitignored. Run `python data/download_data.py` to download the Kaggle dataset.

---

## Evaluation Results

Golden set: 15 hand-labeled Q&A pairs (10 from Apple 10-K, 5 from financial news).

| Config | Chunk Size / Overlap | Retrieval | Ret. Accuracy | Token Overlap |
|---|---|---|---|---|
| Semantic-512 | 512 / 64 | Semantic | 53.3% | 21.2% |
| BM25-512 | 512 / 0 | BM25 | 53.3% | 28.8% |
| Semantic-256 | 256 / 0 | Semantic | 53.3% | 21.9% |
| **Hybrid-512** | **512 / 64** | **Hybrid** | **66.7%** | **35.8%** |

**Key finding:** Hybrid retrieval outperforms all other configurations. News article questions had 0% retrieval accuracy across non-hybrid configs due to document-length imbalance — the Apple 10-K dominates the embedding space.

---

## Setup

```bash
git clone https://github.com/ShawnaLiuuu/DS593_Final_Project.git
cd DS593_Final_Project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

echo "OPENAI_API_KEY=your_key_here" > .env

# Download data
python data/download_data.py
```

---

## Usage

```bash
# Ingest documents
python src/ingest.py --input data/raw/ --output data/processed/documents.json

# Run Streamlit UI
streamlit run app.py

# Run a single query
python src/pipeline.py --query "What was Apple's total net sales for fiscal year 2025?" --config configs/config.yaml

# Run baseline (no RAG)
python src/pipeline.py --query "..." --config configs/config.yaml --baseline

# Run full evaluation
python eval/evaluate.py --golden eval/golden_set.json --config configs/config.yaml --output eval/results.json
```

---

## Ethical Considerations

Financial documents are sensitive; incorrect outputs could affect real decisions. We document hallucination risks, retrieval failure cases, and data privacy concerns around sending document context to external APIs. See the full report for details.