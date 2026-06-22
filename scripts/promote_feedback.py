"""
Promote high-quality user feedback into the eval set.

Reads feedback.jsonl, filters for entries with rating=1 (thumbs up)
that have a corrected_answer or expected_chunks, and adds them as
new eval entries. This grows the eval set over time, catching edge
cases the original 35 hand-written questions missed.

Usage: python scripts/promote_feedback.py [--dry-run]

Run this periodically (e.g., weekly) to grow the eval set.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_PATH = ROOT / "data" / "feedback" / "feedback.jsonl"
EVAL_SET_PATH = ROOT / "eval" / "eval_set.json"


def load_feedback() -> list[dict]:
    if not FEEDBACK_PATH.exists():
        return []
    return [json.loads(line) for line in FEEDBACK_PATH.open(encoding="utf-8")]


def load_eval_set() -> dict:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def is_promotable(entry: dict) -> bool:
    """A feedback entry is promotable if it has useful signal."""
    if entry["rating"] == -1 and entry.get("corrected_answer"):
        return True
    if entry["rating"] == 1 and entry.get("expected_chunks"):
        return True
    if entry.get("corrected_answer") and entry.get("expected_chunks"):
        return True
    return False


def feedback_to_eval_entry(fb: dict, existing_ids: set) -> dict | None:
    """Convert a feedback entry to an eval set entry."""
    base_id = f"fb_{fb['feedback_id']}"
    if base_id in existing_ids:
        return None

    answer = fb.get("corrected_answer") or fb["answer"]
    chunks = fb.get("expected_chunks", [])

    return {
        "id": base_id,
        "question": fb["question"],
        "expected_chunks": chunks,
        "expected_answer": answer,
        "category": "user-feedback",
        "difficulty": "medium",
    }


def main():
    parser = argparse.ArgumentParser(description="Promote feedback to eval set")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be added without modifying files")
    args = parser.parse_args()

    feedback = load_feedback()
    if not feedback:
        print("No feedback found.")
        return

    eval_data = load_eval_set()
    existing_ids = {e["id"] for e in eval_data["entries"]}
    existing_questions = {e["question"] for e in eval_data["entries"]}

    promotable = [fb for fb in feedback if is_promotable(fb)]
    print(f"Total feedback: {len(feedback)}")
    print(f"Promotable: {len(promotable)}")

    new_entries = []
    for fb in promotable:
        if fb["question"] in existing_questions:
            print(f"  SKIP (duplicate question): {fb['question'][:60]}")
            continue
        entry = feedback_to_eval_entry(fb, existing_ids)
        if entry:
            new_entries.append(entry)
            print(f"  ADD: [{entry['id']}] {entry['question'][:60]}")

    if not new_entries:
        print("\nNo new entries to add.")
        return

    if args.dry_run:
        print(f"\nDry run: would add {len(new_entries)} entries. Run without --dry-run to apply.")
        return

    eval_data["entries"].extend(new_entries)
    EVAL_SET_PATH.write_text(
        json.dumps(eval_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nAdded {len(new_entries)} entries. Eval set now has {len(eval_data['entries'])} entries.")


if __name__ == "__main__":
    main()
