"""
app.py
Streamlit UI for the RAG Financial Document Q&A System.
Run with: streamlit run app.py
"""

import sys
import json
import yaml
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest import ingest_directory
from chunker import chunk_all
from embedder import Embedder
from retriever import Retriever
from generator import Generator

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinRAG — Financial Document Q&A",
    page_icon="📈",
    layout="wide"
)

# ── Load config ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

# ── Build pipeline (cached so it only runs once) ──────────────────────────────
@st.cache_resource
def build_pipeline(method, chunk_size, overlap):
    config = load_config()
    processed = Path("data/processed/documents.json")

    if processed.exists():
        with open(processed) as f:
            documents = json.load(f)
    else:
        with st.spinner("Ingesting documents..."):
            documents = ingest_directory(config["paths"]["raw_data"])
            Path("data/processed").mkdir(parents=True, exist_ok=True)
            with open(processed, "w") as f:
                json.dump(documents, f)

    chunks = chunk_all(documents, strategy="overlap",
                       chunk_size=chunk_size, overlap=overlap)

    embedder = None
    if method in ("semantic", "hybrid"):
        embedder = Embedder(config["embedding"]["model"])

    retriever = Retriever(
        chunks=chunks,
        method=method,
        embedder=embedder,
        hybrid_alpha=config["retrieval"]["hybrid_alpha"]
    )
    generator = Generator(
        model=config["generation"]["model"],
        temperature=config["generation"]["temperature"],
        max_tokens=config["generation"]["max_tokens"],
        prompt_strategy=config["generation"]["prompt_strategy"]
    )
    return retriever, generator

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    retrieval_method = st.selectbox(
        "Retrieval Method",
        ["hybrid", "semantic", "bm25"],
        index=0
    )
    chunk_size = st.slider("Chunk Size (tokens)", 128, 1024, 512, step=128)
    overlap = st.slider("Chunk Overlap (tokens)", 0, 256, 64, step=32)
    top_k = st.slider("Top-K Chunks", 1, 10, 5)
    prompt_strategy = st.selectbox("Prompt Strategy", ["base", "structured"], index=1)
    show_baseline = st.checkbox("Also run baseline (no RAG)", value=False)
    show_chunks = st.checkbox("Show retrieved chunks", value=True)

    st.markdown("---")
    st.markdown("**About**")
    st.markdown("RAG system for financial document Q&A. Supports SEC 10-K filings and financial news.")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📈 FinRAG — Financial Document Q&A")
st.markdown("Ask questions about ingested financial documents including the **Apple 10-K (FY2025)** and **financial news articles**.")

# Load pipeline
with st.spinner("Loading pipeline... (first load may take a few minutes)"):
    retriever, generator = build_pipeline(retrieval_method, chunk_size, overlap)
    generator.prompt_strategy = prompt_strategy
    generator.prompt_template = generator.prompt_template  # refresh

st.success("Pipeline ready!", icon="✅")
st.markdown("---")

# Query input
query = st.text_input(
    "Ask a question about the financial documents:",
    placeholder="e.g. What was Apple's total net sales for fiscal year 2025?"
)

example_questions = [
    "What was Apple's total net sales for fiscal year 2025?",
    "What are Apple's primary reportable segments?",
    "What were Apple's main supply chain risk factors?",
    "What caused US export prices to fall in June 2023?",
    "Why did Bitcoin climb above $31,000 in mid-July?",
]
st.markdown("**Example questions:**")
cols = st.columns(len(example_questions))
for i, eq in enumerate(example_questions):
    if cols[i].button(eq[:40] + "...", key=f"ex_{i}"):
        query = eq

if query:
    st.markdown("---")
    col1, col2 = st.columns([1, 1] if show_baseline else [1, 1])

    # RAG answer
    with col1:
        st.subheader(f"🔍 RAG Answer ({retrieval_method})")
        with st.spinner("Retrieving and generating..."):
            retrieved = retriever.retrieve(query, top_k=top_k)
            rag_answer = generator.generate(query, retrieved=retrieved)
        st.markdown(rag_answer)

        if show_chunks:
            with st.expander(f"📄 Top {top_k} Retrieved Chunks"):
                for i, (chunk, score) in enumerate(retrieved, 1):
                    st.markdown(f"**[{i}] {chunk.doc_id}** — score: `{score:.3f}`")
                    st.markdown(f"> {chunk.text[:400]}...")
                    st.markdown("---")

    # Baseline answer
    if show_baseline:
        with col2:
            st.subheader("🤖 Baseline (No RAG)")
            with st.spinner("Generating baseline answer..."):
                baseline_answer = generator.generate(query, retrieved=None)
            st.markdown(baseline_answer)

    # Comparison note
    if show_baseline:
        st.markdown("---")
        st.info("💡 **RAG vs Baseline:** The RAG answer is grounded in retrieved document chunks. The baseline relies only on the LLM's parametric knowledge, which may be outdated or hallucinated for specific financial figures.")
