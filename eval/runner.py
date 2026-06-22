"""
Eval runner: scores a retrieval function against the eval set.

The key abstraction is the retrieval function signature:
    retrieve_fn(question: str, k: int) -> list[str]
It takes a question and returns k chunk_ids, ranked by relevance.
Any retrieval implementation (BM25, vector, hybrid) just needs to
conform to this signature.

The runner loads the eval set, calls retrieve_fn for each question,
scores each result with the metrics from scoring.py, and produces
both per-question results and aggregate statistics.
"""

import json
import statistics
from pathlib import Path
from typing import Callable

from eval.scoring import score_single

RetrieveFn = Callable[[str, int], list[str]]

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict]:
    """Load and validate the eval set."""
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("version") == "1.0", (
        f"Unsupported eval set version: {data.get('version')}"
    )
    entries = data["entries"]
    required_keys = {"id", "question", "expected_chunks", "category", "difficulty"}
    for entry in entries:
        missing = required_keys - entry.keys()
        assert not missing, f"Entry {entry.get('id', '?')} missing keys: {missing}"
    return entries


def run_eval(
    retrieve_fn: RetrieveFn,
    k: int = 5,
    k_values: list[int] | None = None,
    categories: list[str] | None = None,
) -> dict:
    """Run the eval set against a retrieval function.

    Args:
        retrieve_fn: callable(question, k) -> ranked list of chunk_ids
        k: how many chunks to retrieve per question
        k_values: which k values to compute recall/precision at
        categories: if set, only run questions in these categories

    Returns:
        dict with "per_question", "aggregate", and "by_category" results
    """
    if k_values is None:
        k_values = [1, 3, 5]

    entries = load_eval_set()
    if categories:
        entries = [e for e in entries if e["category"] in categories]

    per_question = []
    for entry in entries:
        retrieved = retrieve_fn(entry["question"], k)
        scores = score_single(retrieved, entry["expected_chunks"], k_values)
        per_question.append({
            "id": entry["id"],
            "question": entry["question"],
            "category": entry["category"],
            "difficulty": entry["difficulty"],
            "expected_chunks": entry["expected_chunks"],
            "retrieved_chunks": retrieved[:k],
            **scores,
        })

    metric_keys = [
        key for key in per_question[0]
        if key.startswith(("recall", "precision", "reciprocal"))
    ]

    aggregate = {
        key: statistics.mean([q[key] for q in per_question])
        for key in metric_keys
    }

    by_category = {}
    for cat in sorted(set(q["category"] for q in per_question)):
        cat_qs = [q for q in per_question if q["category"] == cat]
        by_category[cat] = {
            "count": len(cat_qs),
            **{key: statistics.mean([q[key] for q in cat_qs]) for key in metric_keys},
        }

    return {
        "per_question": per_question,
        "aggregate": aggregate,
        "by_category": by_category,
    }


def print_scorecard(results: dict) -> None:
    """Pretty-print eval results as a terminal scorecard."""
    agg = results["aggregate"]
    n = len(results["per_question"])

    print("=" * 60)
    print("EVAL SCORECARD")
    print("=" * 60)
    print(f"  Questions evaluated: {n}")
    print()

    for key in sorted(agg):
        print(f"  {key:<20} {agg[key]:.3f}")

    print(f"\n  By category:")
    print(f"  {'Category':<16} {'Count':>5}  {'Recall@5':>9}  {'MRR':>9}")
    print(f"  {'-'*16} {'-'*5}  {'-'*9}  {'-'*9}")
    for cat, metrics in results["by_category"].items():
        r5 = metrics.get("recall@5", 0)
        mrr = metrics.get("reciprocal_rank", 0)
        print(f"  {cat:<16} {metrics['count']:>5}  {r5:>9.3f}  {mrr:>9.3f}")

    failures = [q for q in results["per_question"] if q.get("recall@5", 0) < 1.0]
    if failures:
        print(f"\n  Missed questions ({len(failures)}):")
        for q in failures:
            print(f"    [{q['id']}] {q['question'][:60]}")
            print(f"      expected:  {q['expected_chunks']}")
            print(f"      got:       {q['retrieved_chunks'][:5]}")
    else:
        print(f"\n  All questions answered correctly (recall@5 = 1.0)")
