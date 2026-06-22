"""
Validate chunks.jsonl: check for duplicates, empty chunks, token counts,
and section coverage. Prints a summary table.

Run: python scripts/validate_chunks.py
"""

import json
import statistics
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"

EXPECTED_SECTIONS = {
    "GEN_3A.1", "GEN_3A.2",
    "GEN_A2.5",
    "COB_15.1", "COB_15.2", "COB_15.3", "COB_15.4",
    "COB_15.5", "COB_15.6", "COB_15.7", "COB_15.8",
    "AML_9.3",
}

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def main():
    if not CHUNKS_PATH.exists():
        print(f"ERROR: {CHUNKS_PATH} not found. Run chunk_rules.py first.")
        return

    chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
    print(f"Total chunks: {len(chunks)}\n")

    # --- Duplicate check ---
    ids = [c["chunk_id"] for c in chunks]
    dupes = set(x for x in ids if ids.count(x) > 1)
    if dupes:
        print(f"FAIL: duplicate chunk_ids: {dupes}")
    else:
        print("OK: no duplicate chunk_ids")

    # --- Empty check ---
    empty = [c["chunk_id"] for c in chunks if not c["text"].strip()]
    if empty:
        print(f"FAIL: empty chunks: {empty}")
    else:
        print("OK: no empty chunks")

    # --- Token counts ---
    token_counts = [(c["chunk_id"], count_tokens(c["text"])) for c in chunks]
    counts = [t for _, t in token_counts]

    print(f"\nToken count distribution:")
    print(f"  min:    {min(counts)}")
    print(f"  median: {statistics.median(counts):.0f}")
    print(f"  mean:   {statistics.mean(counts):.0f}")
    print(f"  max:    {max(counts)}")

    # Flag outliers
    small = [(cid, t) for cid, t in token_counts if t < 50]
    large = [(cid, t) for cid, t in token_counts if t > 800]
    if small:
        print(f"\n  WARNING: {len(small)} chunks under 50 tokens:")
        for cid, t in small:
            print(f"    {cid}: {t} tokens")
    if large:
        print(f"\n  WARNING: {len(large)} chunks over 800 tokens:")
        for cid, t in large:
            print(f"    {cid}: {t} tokens")

    # --- Section coverage ---
    sections_found = set()
    for c in chunks:
        module = c["module"]
        section = c["section"]
        sections_found.add(f"{module}_{section}")

    print(f"\nSections covered: {sorted(sections_found)}")
    missing = EXPECTED_SECTIONS - sections_found
    if missing:
        print(f"WARNING: expected sections not found: {missing}")
    else:
        print("OK: all expected sections covered")

    # --- Per-section summary ---
    print(f"\nPer-section breakdown:")
    print(f"  {'Section':<12} {'Chunks':>6} {'Min tok':>8} {'Max tok':>8} {'Guidance':>9}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*9}")

    section_groups: dict[str, list] = {}
    for c, (_, tok) in zip(chunks, token_counts):
        key = f"{c['module']}_{c['section']}"
        section_groups.setdefault(key, []).append((c, tok))

    for section_key in sorted(section_groups):
        items = section_groups[section_key]
        toks = [t for _, t in items]
        guidance_count = sum(1 for c, _ in items if c["has_guidance"])
        print(
            f"  {section_key:<12} {len(items):>6} {min(toks):>8} {max(toks):>8} {guidance_count:>9}"
        )


if __name__ == "__main__":
    main()
