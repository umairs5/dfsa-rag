"""
Scrape DFSA Rulebook sections (GEN 3A, COB 15, AML 9.3A/9.3B) from
the Thomson Reuters-hosted rulebook at dfsaen.thomsonreuters.com.

Two-pass approach:
  1. Discovery: starting from section index pages, crawl to find all leaf
     pages that contain actual rule text.
  2. Extraction: fetch each leaf page and extract the regulatory text.

Output: one markdown file per section in data/raw/, named by section ID
(e.g., GEN_3A.2.1.md).
"""

import re
import time
import logging
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://dfsaen.thomsonreuters.com"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DELAY = 1.5  # seconds between requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Seed pages: section index pages we start discovery from.
SEED_PAGES = {
    "GEN_3A": "/rulebook/gen-3a-guidance",
    "GEN_3A.1": "/rulebook/gen-3a1-definitions",
    "GEN_3A.2": "/rulebook/gen-3a2-prohibitions-relating-crypto-tokens",
    "GEN_App2.5": "/rulebook/gen-a25-definitions-relating-crypto-tokens",
    "COB_15": "/rulebook/cob-15-additional-requirements-firms-providing-financial-services-relating-crypto-tokens",
    "COB_15.1": "/rulebook/cob-151-application",
    "COB_15.2": "/rulebook/cob-152-operating-mtf-crypto-tokens-which-permits-direct-access",
    "COB_15.3": "/rulebook/cob-153-disclosure-information-about-crypto-tokens-mtf",
    "COB_15.4": "/rulebook/cob-154-requirements-providing-custody-crypto-tokens",
    "COB_15.5": "/rulebook/cob-155-provision-information",
    "COB_15.6": "/rulebook/cob-156-general-requirements-relating-crypto-tokens-and-crypto-token-derivatives",
    "COB_15.7": "/rulebook/cob-157-technology-and-governance-requirements",
    "COB_15.8": "/rulebook/cob-158-technology-audit-reports",
    "AML_9.3A": "/rulebook/aml-93a-additional-requirements-crypto-token-transfers",
    "AML_9.3B": "/rulebook/aml-93b-additional-requirements-nft-and-utility-token-transfers",
}

session = requests.Session()
session.headers.update(HEADERS)


def fetch(path: str) -> BeautifulSoup | None:
    url = urljoin(BASE_URL, path)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        time.sleep(DELAY)
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        log.warning("Failed to fetch %s: %s", url, e)
        return None


def extract_text(soup: BeautifulSoup) -> str:
    """Extract the main regulatory text content from a rulebook page."""
    # The Thomson Reuters rulebook uses a main content area.
    # Try several selectors to find the right content block.
    content = (
        soup.find("div", class_="field-item")
        or soup.find("div", class_="node-content")
        or soup.find("article")
        or soup.find("div", id="content")
        or soup.find("main")
    )
    if content is None:
        content = soup.body

    if content is None:
        return ""

    # Remove nav, menus, footers, sidebars
    for tag in content.find_all(["nav", "footer", "script", "style", "noscript"]):
        tag.decompose()

    # Get text with some structure preserved
    lines = []
    for element in content.descendants:
        if element.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(element.name[1])
            lines.append(f"\n{'#' * level} {element.get_text(strip=True)}\n")
        elif element.name == "p":
            text = element.get_text(strip=True)
            if text:
                lines.append(text + "\n")
        elif element.name == "li":
            text = element.get_text(strip=True)
            if text:
                lines.append(f"- {text}\n")
        elif element.name == "td":
            text = element.get_text(strip=True)
            if text:
                lines.append(f"| {text} ")
        elif element.name == "tr":
            lines.append("|\n")

    result = "\n".join(lines)
    # Collapse excessive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def get_page_title(soup: BeautifulSoup) -> str:
    title_tag = soup.find("h1")
    if title_tag:
        return title_tag.get_text(strip=True)
    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True).split("|")[0].strip()
    return "Untitled"


def discover_child_links(soup: BeautifulSoup, parent_path: str) -> list[str]:
    """Find links to child/subsection pages from a section index page."""
    links = []
    seen = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Only follow links within the rulebook
        if not href.startswith("/rulebook/"):
            continue
        # Skip the parent page itself
        if href == parent_path:
            continue
        # Skip module-level table-of-contents pages
        if re.match(r"/rulebook/(general-module|conduct-business-module|anti-money-laundering)", href):
            continue
        if href not in seen:
            seen.add(href)
            links.append(href)
    return links


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\s\-.]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:100]


def scrape_section(section_id: str, seed_path: str) -> list[dict]:
    """Scrape a section: discover children from seed, then extract text."""
    log.info("=== Discovering pages for %s ===", section_id)
    pages = []

    soup = fetch(seed_path)
    if soup is None:
        return pages

    # Check if this seed page itself has content (leaf page)
    text = extract_text(soup)
    title = get_page_title(soup)
    if text and len(text) > 100:
        pages.append({"path": seed_path, "title": title, "text": text, "section_id": section_id})

    # Discover child pages
    child_paths = discover_child_links(soup, seed_path)
    log.info("  Found %d child links from %s", len(child_paths), seed_path)

    for child_path in child_paths:
        child_soup = fetch(child_path)
        if child_soup is None:
            continue
        child_title = get_page_title(child_soup)
        child_text = extract_text(child_soup)
        if child_text and len(child_text) > 50:
            pages.append({
                "path": child_path,
                "title": child_title,
                "text": child_text,
                "section_id": section_id,
            })
            # Also check for grandchildren (some pages nest further)
            grandchild_paths = discover_child_links(child_soup, child_path)
            for gc_path in grandchild_paths:
                if any(p["path"] == gc_path for p in pages):
                    continue
                gc_soup = fetch(gc_path)
                if gc_soup is None:
                    continue
                gc_title = get_page_title(gc_soup)
                gc_text = extract_text(gc_soup)
                if gc_text and len(gc_text) > 50:
                    pages.append({
                        "path": gc_path,
                        "title": gc_title,
                        "text": gc_text,
                        "section_id": section_id,
                    })

    return pages


def save_page(page: dict, output_dir: Path) -> Path:
    filename = sanitize_filename(page["title"]) + ".md"
    filepath = output_dir / filename
    content = f"---\nsource: {BASE_URL}{page['path']}\nsection: {page['section_id']}\ntitle: \"{page['title']}\"\n---\n\n# {page['title']}\n\n{page['text']}\n"
    filepath.write_text(content, encoding="utf-8")
    return filepath


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pages = []
    seen_paths = set()

    for section_id, seed_path in SEED_PAGES.items():
        pages = scrape_section(section_id, seed_path)
        for page in pages:
            if page["path"] not in seen_paths:
                seen_paths.add(page["path"])
                all_pages.append(page)

    log.info("\n=== Saving %d pages ===", len(all_pages))
    for page in all_pages:
        filepath = save_page(page, OUTPUT_DIR)
        log.info("  Saved: %s", filepath.name)

    log.info("\nDone. %d files written to %s", len(all_pages), OUTPUT_DIR)


if __name__ == "__main__":
    main()
