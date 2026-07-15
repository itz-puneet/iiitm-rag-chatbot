"""
Crawl a college website, find every link ending in .pdf, and download them.

Designed for building a RAG knowledge base from https://www.iiitm.ac.in

Usage:
    python download_pdfs.py                       # crawl default site, 2 levels deep
    python download_pdfs.py --url <URL> --depth 3 --out ./pdfs

Dependencies:
    pip install requests beautifulsoup4
"""

import argparse
import os
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

# A real User-Agent avoids being blocked by many servers.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

DEFAULT_URL = "https://www.iiitm.ac.in/index.php/en/"


def is_same_domain(url: str, root_netloc: str) -> bool:
    """Keep the crawl inside the college's own domain."""
    return urlparse(url).netloc in ("", root_netloc)


def safe_filename(url: str) -> str:
    """Turn a PDF URL into a safe, unique local filename."""
    name = unquote(os.path.basename(urlparse(url).path))
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    # Strip characters that are illegal in Windows filenames.
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name or "download.pdf"


def download_pdf(session: requests.Session, url: str, out_dir: str) -> None:
    """Stream a PDF to disk, skipping files already downloaded."""
    filename = safe_filename(url)
    path = os.path.join(out_dir, filename)

    # Avoid overwriting: if two PDFs share a name, add a counter.
    counter = 1
    base, ext = os.path.splitext(path)
    while os.path.exists(path):
        # Same URL already fetched? Skip. Otherwise disambiguate.
        path = f"{base}_{counter}{ext}"
        counter += 1

    try:
        with session.get(url, headers=HEADERS, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            # Guard against HTML error pages served with a .pdf URL.
            ctype = resp.headers.get("Content-Type", "").lower()
            if "pdf" not in ctype and "octet-stream" not in ctype:
                print(f"  ! Skipping (not a PDF, got {ctype}): {url}")
                return
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        size_kb = os.path.getsize(path) / 1024
        print(f"  ✓ Saved {os.path.basename(path)} ({size_kb:.0f} KB)")
    except requests.RequestException as e:
        print(f"  ! Failed to download {url}: {e}")


def crawl(start_url: str, out_dir: str, max_depth: int, delay: float) -> None:
    os.makedirs(out_dir, exist_ok=True)
    root_netloc = urlparse(start_url).netloc

    session = requests.Session()
    visited_pages = set()
    downloaded_pdfs = set()

    # BFS queue of (url, depth).
    queue = deque([(start_url, 0)])

    while queue:
        url, depth = queue.popleft()
        if url in visited_pages or depth > max_depth:
            continue
        visited_pages.add(url)

        print(f"[depth {depth}] Crawling: {url}")
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  ! Could not fetch page: {e}")
            continue

        # Only parse HTML pages for links.
        if "text/html" not in resp.headers.get("Content-Type", "").lower():
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"].strip())
            link, _, _ = link.partition("#")  # drop fragments

            if link.lower().endswith(".pdf"):
                if link not in downloaded_pdfs:
                    downloaded_pdfs.add(link)
                    print(f"  → Found PDF: {link}")
                    download_pdf(session, link, out_dir)
            elif (
                depth < max_depth
                and is_same_domain(link, root_netloc)
                and link not in visited_pages
                and link.startswith("http")
            ):
                queue.append((link, depth + 1))

        time.sleep(delay)  # be polite; don't hammer the server

    print(f"\nDone. {len(downloaded_pdfs)} unique PDF link(s) found. Saved to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl a site and download all PDFs.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Start URL")
    parser.add_argument("--out", default="./pdfs", help="Output folder")
    parser.add_argument("--depth", type=int, default=2, help="Max crawl depth")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests")
    args = parser.parse_args()

    crawl(args.url, args.out, args.depth, args.delay)


if __name__ == "__main__":
    main()
