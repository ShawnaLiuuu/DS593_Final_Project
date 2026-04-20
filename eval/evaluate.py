"""
evaluate.py
Evaluate retrieval accuracy and answer quality against a golden Q&A set.

Golden set format (eval/golden_set.json):
[
  {
    "id": "q1",
    "question": "Why did revenue decrease this quarter?",
    "answer": "Revenue decreased due to ...",
    "relevant_doc_ids": ["apple_10k_2023"]   // for retrieval accuracy
  },
  ...
]

Usage:
    python eval/evaluate.py --golden eval/golden_set.json --config configs/config.yaml
"""

import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import load_config, build_pipeline, run_query


def retrieval_accuracy(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    """1 if any relevant doc appears in retrieved chunks, 0 otherwise."""
    return float(any(doc_id in retrieved_doc_ids for doc_id in relevant_doc_ids))


def score_answer(predicted: str, reference: str) -> dict:
    """
    Simple scoring:
      - exact_match: 1 if normalized strings match
      - token_overlap: Jaccard similarity of word sets (rough proxy for F1)
    For real evaluation, use manual rubric scores (0–3) in golden_set.json.
    """
    pred_tokens = set(predicted.lower().split())
    ref_tokens = set(reference.lower().split())
    intersection = pred_tokens & ref_tokens
    union = pred_tokens | ref_tokens
    jaccard = len(intersection) / len(union) if union else 0.0
    exact = float(predicted.strip().lower() == reference.strip().lower())
    return {"exact_match": exact, "token_overlap": jaccard}


def evaluate(golden: list[dict], retriever, generator, top_k: int) -> dict:
    results = []

    for item in golden:
        result = run_query(item["question"], retriever, generator, top_k)
        retrieved_doc_ids = [r["doc_id"] for r in result["retrieved"]]
        ret_acc = retrieval_accuracy(retrieved_doc_ids, item.get("relevant_doc_ids", []))
        ans_scores = score_answer(result["answer"], item["answer"])

        results.append({
            "id": item["id"],
            "question": item["question"],
            "reference_answer": item["answer"],
            "predicted_answer": result["answer"],
            "retrieval_accuracy": ret_acc,
            **ans_scores
        })
        print(f"  [{item['id']}] ret_acc={ret_acc:.0f}  token_overlap={ans_scores['token_overlap']:.2f}")

    avg_ret = sum(r["retrieval_accuracy"] for r in results) / len(results)
    avg_tok = sum(r["token_overlap"] for r in results) / len(results)
    summary = {"n": len(results), "avg_retrieval_accuracy": avg_ret, "avg_token_overlap": avg_tok}
    print(f"\nSummary: {summary}")
    return {"summary": summary, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="eval/golden_set.json")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="eval/results.json")
    args = parser.parse_args()

    with open(args.golden) as f:
        golden = json.load(f)

    config = load_config(args.config)
    retriever, generator, top_k = build_pipeline(config)

    eval_results = evaluate(golden, retriever, generator, top_k)

    with open(args.output, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nResults saved to {args.output}")
