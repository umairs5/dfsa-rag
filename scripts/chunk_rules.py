"""
Split extracted text into rule-level chunks with metadata.

Reads raw text files from data/intermediate/, detects rule boundaries
and guidance blocks, attaches metadata (section, page, cross-refs),
and writes JSONL to data/processed/chunks.jsonl.

Run: python scripts/chunk_rules.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INTERMEDIATE_DIR = ROOT / "data" / "intermediate"
OUT_DIR = ROOT / "data" / "processed"

# --- Configuration per source file ---
# Each entry: (filename, module, rule_prefix_filter, section_titles)
#
# rule_prefix_filter: only keep rules whose ID starts with this prefix.
#   This avoids false positives from cross-references like "Rule 2.13.1"
#   appearing at the start of a line due to PDF line breaks.
#
# section_titles: maps section number to its title, used to build the
#   context_header breadcrumb. Manually extracted from the PDFs.

SOURCES = [
    {
        "file": "gen_3a_raw.txt",
        "module": "GEN",
        "source_pdf": "General Module.pdf",
        "rule_prefix": "3A.",
        "section_titles": {
            "3A": "Crypto Token Requirements",
            "3A.1": "Definitions",
            "3A.2": "Prohibitions relating to Crypto Tokens",
        },
    },
    {
        "file": "gen_app2_a25_raw.txt",
        "module": "GEN",
        "source_pdf": "General Module.pdf",
        "rule_prefix": "A2.5.",
        "section_titles": {
            "A2.5": "Definitions relating to Crypto Tokens",
        },
    },
    {
        "file": "cob_15_raw.txt",
        "module": "COB",
        "source_pdf": "Conduct of Business Module.pdf",
        "rule_prefix": "15.",
        "section_titles": {
            "15": "Additional Requirements for Firms Providing Financial Services Relating to Crypto Tokens",
            "15.1": "Application",
            "15.2": "Operating a MTF for Crypto Tokens which permits Direct Access",
            "15.3": "Disclosure of information about Crypto Tokens on a MTF",
            "15.4": "Requirements for Providing Custody of Crypto Tokens",
            "15.5": "Provision of Information",
            "15.6": "General requirements relating to Crypto Tokens and Crypto Token Derivatives",
            "15.7": "Technology and Governance Requirements",
            "15.8": "Technology Audit Reports",
        },
    },
    {
        "file": "aml_9_3_raw.txt",
        "module": "AML",
        "source_pdf": "Anti-Money Laundering, Counter-Terrorist.pdf",
        "rule_prefix": "9.3.",
        "section_titles": {
            "9.3A": "Additional requirements for Crypto Token transfers",
            "9.3B": "Additional requirements for NFT and Utility Token transfers",
        },
    },
]

# Regex to detect rule boundaries. Matches patterns like:
#   3A.2.1    15.6.5    A2.5.1    3A.2.1A    15.6.10
# Must be at the start of a line (after optional whitespace).
RULE_RE = re.compile(
    r"^((?:A\d+|\d+[A-Z]?)\.\d+\.\d+[A-Z]?)\s",
    re.MULTILINE,
)

# Regex to detect section headers like "3A.2 Prohibitions..." or "15.1 Application"
SECTION_HEADER_RE = re.compile(
    r"^((?:A\d+|\d+[A-Z]?)\.\d+)\s+[A-Z]",
    re.MULTILINE,
)

# Regex to detect "Guidance" blocks
GUIDANCE_RE = re.compile(r"^\s*Guidance\s*$", re.MULTILINE)

# Regex to extract cross-references like "Rule 3A.2.1", "COB 15.6", "section A2.5"
XREF_RE = re.compile(
    r"(?:Rule|section|chapter|GEN|COB|AML|GLO)\s+"
    r"([\dA-Z]+[\.\d]*[A-Za-z]*(?:\(\d+\)(?:\([a-z]\))?)?)",
    re.IGNORECASE,
)

PAGE_MARKER_RE = re.compile(r"<<<PAGE (\d+)>>>")


def get_current_page(text: str, pos: int) -> int | None:
    """Find the most recent page marker before position `pos`."""
    best = None
    for m in PAGE_MARKER_RE.finditer(text):
        if m.start() <= pos:
            best = int(m.group(1))
        else:
            break
    return best


def build_context_header(module: str, rule_id: str, section_titles: dict) -> str:
    """Build a breadcrumb like 'GEN > 3A Crypto Token Requirements > 3A.2 Prohibitions'."""
    parts = [module]
    # Walk from broadest section to narrowest
    # e.g., for rule "3A.2.1", check "3A" then "3A.2"
    tokens = rule_id.split(".")
    for i in range(1, len(tokens)):
        section_key = ".".join(tokens[:i])
        if section_key in section_titles:
            parts.append(f"{section_key} {section_titles[section_key]}")
    return " > ".join(parts)


def extract_cross_references(text: str) -> list[str]:
    """Extract unique cross-references from chunk text."""
    refs = []
    seen = set()
    for m in XREF_RE.finditer(text):
        ref = m.group(1)
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def chunk_source(source: dict) -> list[dict]:
    """Parse a single raw text file into rule-level chunks."""
    filepath = INTERMEDIATE_DIR / source["file"]
    if not filepath.exists():
        print(f"SKIP: {filepath} not found")
        return []

    text = filepath.read_text(encoding="utf-8")
    module = source["module"]
    prefix = source["rule_prefix"]
    section_titles = source["section_titles"]

    # Find all rule boundary positions.
    # Filter out false positives: if the line before the match ends with
    # "Rule" or "rule", this is a cross-reference mid-sentence, not a
    # genuine rule boundary. Also skip duplicate rule IDs (keep first).
    boundaries = []
    seen_ids = set()
    for m in RULE_RE.finditer(text):
        rule_id = m.group(1)
        if not rule_id.startswith(prefix):
            continue
        # Check if the previous non-blank line ends with "Rule" (cross-ref)
        preceding = text[:m.start()].rstrip()
        if preceding.endswith("Rule") or preceding.endswith("rule"):
            continue
        if rule_id in seen_ids:
            continue
        seen_ids.add(rule_id)
        boundaries.append((m.start(), rule_id))

    if not boundaries:
        print(f"  WARNING: no rules found in {source['file']}")
        return []

    chunks = []
    for i, (start, rule_id) in enumerate(boundaries):
        # Chunk text runs from this rule boundary to the next (or end of file)
        if i + 1 < len(boundaries):
            end = boundaries[i + 1][0]
        else:
            end = len(text)

        chunk_text = text[start:end].strip()

        # Remove page markers from the chunk text (they're metadata, not content)
        clean_text = PAGE_MARKER_RE.sub("", chunk_text).strip()
        # Collapse runs of blank lines left by marker removal
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

        # Detect if this chunk contains a Guidance block
        has_guidance = bool(GUIDANCE_RE.search(clean_text))

        # Get page range
        page_start = get_current_page(text, start)
        page_end = get_current_page(text, end - 1) or page_start

        # Build context header
        context_header = build_context_header(module, rule_id, section_titles)

        # Extract cross-references
        cross_refs = extract_cross_references(clean_text)

        # Determine section from rule_id (e.g., "3A.2" from "3A.2.1")
        section = ".".join(rule_id.split(".")[:2])

        chunk = {
            "chunk_id": f"{module}_{rule_id}",
            "module": module,
            "section": section,
            "section_title": section_titles.get(section, ""),
            "rule_id": rule_id,
            "has_guidance": has_guidance,
            "page_start": page_start,
            "page_end": page_end,
            "source_pdf": source["source_pdf"],
            "cross_references": cross_refs,
            "context_header": context_header,
            "text": clean_text,
        }
        chunks.append(chunk)

    return chunks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "chunks.jsonl"

    all_chunks = []
    for source in SOURCES:
        print(f"Processing {source['file']}...")
        chunks = chunk_source(source)
        print(f"  -> {len(chunks)} chunks")
        all_chunks.append(chunks)

    # Flatten and check for duplicate chunk_ids
    flat = [c for group in all_chunks for c in group]
    ids = [c["chunk_id"] for c in flat]
    dupes = [x for x in ids if ids.count(x) > 1]
    if dupes:
        print(f"\nWARNING: duplicate chunk_ids: {set(dupes)}")

    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in flat:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(flat)} chunks to {out_path}")


if __name__ == "__main__":
    main()
