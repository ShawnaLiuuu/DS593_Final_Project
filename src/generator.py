"""
generator.py
LLM-based answer generation using retrieved context.
Supports two prompt strategies: "base" and "structured".
"""
from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI
from chunker import Chunk


BASE_PROMPT = """\
Use the following document excerpts to answer the question.
If the answer is not contained in the excerpts, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
Answer:"""

STRUCTURED_PROMPT = """\
You are a financial analyst assistant. Answer the question using ONLY the provided document excerpts.
For each claim in your answer, cite the relevant excerpt (e.g., [1], [2]).
If the answer cannot be found in the excerpts, explicitly say so.

Excerpts:
{context}

Question: {question}

Answer (with citations):"""


def format_context(chunks_with_scores: list[tuple[Chunk, float]]) -> str:
    parts = []
    for i, (chunk, score) in enumerate(chunks_with_scores, 1):
        parts.append(f"[{i}] (source: {chunk.doc_id}, score: {score:.3f})\n{chunk.text}")
    return "\n\n".join(parts)


class Generator:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0,
                 max_tokens: int = 512, prompt_strategy: str = "base"):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = BASE_PROMPT if prompt_strategy == "base" else STRUCTURED_PROMPT

    def generate(self, query: str,
                 retrieved: list[tuple[Chunk, float]] | None = None) -> str:
        """
        retrieved=None → no-RAG baseline (pure LLM)
        retrieved=[...] → RAG mode
        """
        if retrieved is None:
            user_msg = f"Answer this question about a financial document: {query}"
        else:
            context = format_context(retrieved)
            user_msg = self.prompt_template.format(context=context, question=query)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": user_msg}]
        )
        return response.choices[0].message.content.strip()
