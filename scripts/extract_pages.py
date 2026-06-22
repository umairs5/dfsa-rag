"""
Extract raw text from the relevant sections of the DFSA rulebook PDFs.

Reads only the in-scope page ranges (GEN 3A, GEN App2 A2.5, COB 15,
AML 9.3A/9.3B), strips repeated headers/footers, inserts page markers,
and writes one .txt file per section into data/intermediate/.

Run: python scripts/extract_pages.py
"""

import re
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "intermediate"

# Each entry: (output_filename, pdf_filename, start_page, end_page)
# Pages are 1-indexed (inclusive) to match what you see in a PDF viewer.
SECTIONS = [
    ("gen_3a_raw.txt", "General Module.pdf", 65, 72),
    ("gen_app2_a25_raw.txt", "General Module.pdf", 176, 186),
    ("cob_15_raw.txt", "Conduct of Business Module.pdf", 188, 203),
    ("aml_9_3_raw.txt", "Anti-Money Laundering, Counter-Terrorist.pdf", 52, 54),
]

# Header/footer pattern: each page starts with the module name, a page
# number, and a version string like "GEN/VER71/01-26". We strip these
# so the downstream chunker sees only rule text.
HEADER_RE = re.compile(
    r"^\s*"
    r"(?:GENERAL \(GEN\)|CONDUCT OF BUSINESS \(COB\)|"
    r"Anti-Money Laundering, Counter-Terrorist Financing and Sanctions Module \(AML\))"
    r"\s*\n"
    r"\s*\d+\s*\n"
    r"\s*(?:GEN|COB|AML)/VER\d+/\d{2}-\d{2}\s*\n",
    re.MULTILINE,
)


def extract_section(pdf_path: Path, page_start: int, page_end: int) -> str:
    """Extract text from a page range, strip headers, insert page markers."""
    doc = fitz.open(pdf_path)
    parts: list[str] = []

    for page_num in range(page_start, page_end + 1):
        page = doc[page_num - 1]  # fitz uses 0-indexed pages
        text = page.get_text("text")

        # Strip the header/footer block
        text = HEADER_RE.sub("", text)

        # Remove leading/trailing whitespace per page
        text = text.strip()

        if text:
            parts.append(f"<<<PAGE {page_num}>>>\n{text}")

    doc.close()
    return "\n\n".join(parts)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for out_name, pdf_name, start, end in SECTIONS:
        pdf_path = RAW_DIR / pdf_name
        if not pdf_path.exists():
            print(f"SKIP: {pdf_path} not found")
            continue

        text = extract_section(pdf_path, start, end)
        out_path = OUT_DIR / out_name
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path.name}: pages {start}-{end}, {len(text):,} chars")


if __name__ == "__main__":
    main()
