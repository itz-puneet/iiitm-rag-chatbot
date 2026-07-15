"""
100% free, fully-local, table-aware PDF -> Markdown extraction for RAG.

No cloud, no API keys, no page limits. Two engines, chosen automatically per file:

  * Digital PDFs  -> pymupdf4llm : converts text + complex tables (fee
                     structures, credit/course requirements) into clean Markdown.
  * Scanned PDFs  -> RapidOCR    : image-only pages are OCR'd (offline, pip-only).

The script auto-detects which PDFs are scanned (near-zero embedded text) and
routes them to OCR. It is RESUMABLE: any PDF whose .md already exists is skipped.

Setup
-----
    pip install pymupdf4llm            # digital extraction (pulls in PyMuPDF)
    pip install rapidocr-onnxruntime   # OCR for scanned PDFs (optional but recommended)

Usage
-----
    python extract_pdfs_local.py                       # ./pdfs -> ./extracted
    python extract_pdfs_local.py --in ./pdfs --out ./extracted
    python extract_pdfs_local.py --no-ocr              # skip scanned files
    python extract_pdfs_local.py --ocr-dpi 250         # sharper OCR (slower)
"""

import argparse
import sys
from pathlib import Path

import pymupdf
import pymupdf4llm

# OCR is optional: if RapidOCR isn't installed, scanned PDFs are reported & skipped.
try:
    from rapidocr_onnxruntime import RapidOCR
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False

# A page averaging fewer than this many characters is treated as a scanned image.
SCANNED_CHARS_PER_PAGE = 90


def is_scanned(pdf_path: Path) -> bool:
    doc = pymupdf.open(pdf_path)
    try:
        pages = doc.page_count or 1
        chars = sum(len(page.get_text()) for page in doc)
    finally:
        doc.close()
    return chars / pages < SCANNED_CHARS_PER_PAGE


def extract_digital(pdf_path: Path) -> str:
    """Text + tables -> Markdown via PyMuPDF4LLM."""
    return pymupdf4llm.to_markdown(str(pdf_path), show_progress=False).strip()


def extract_ocr(pdf_path: Path, ocr: "RapidOCR", dpi: int) -> str:
    """OCR each page image and join with page markers."""
    doc = pymupdf.open(pdf_path)
    parts = []
    try:
        for i, page in enumerate(doc, start=1):
            png = page.get_pixmap(dpi=dpi).tobytes("png")
            result, _ = ocr(png)
            text = "\n".join(line[1] for line in (result or [])).strip()
            if text:
                parts.append(f"<!-- page {i} (OCR) -->\n\n{text}")
    finally:
        doc.close()
    return "\n\n---\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Free local PDF -> Markdown for RAG.")
    ap.add_argument("--in", dest="in_dir", default="./pdfs")
    ap.add_argument("--out", dest="out_dir", default="./extracted")
    ap.add_argument("--no-ocr", action="store_true", help="skip scanned PDFs entirely")
    ap.add_argument("--ocr-dpi", type=int, default=200, help="render DPI for OCR")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(in_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {in_dir.resolve()}")

    use_ocr = _HAS_OCR and not args.no_ocr
    ocr = RapidOCR() if use_ocr else None          # init once (model load is costly)
    if not _HAS_OCR and not args.no_ocr:
        print("NOTE: rapidocr-onnxruntime not installed -> scanned PDFs will be skipped.")
        print("      Install it with:  pip install rapidocr-onnxruntime\n")

    tally = {"digital": 0, "ocr": 0, "skip": 0, "scanned_skipped": 0, "fail": 0}
    scanned_skipped, failures = [], []

    for n, pdf in enumerate(pdfs, start=1):
        out_file = out_dir / (pdf.stem + ".md")
        if out_file.exists() and out_file.stat().st_size > 0:
            tally["skip"] += 1
            print(f"  [{n:>3}/{len(pdfs)}] skip     {pdf.name}")
            continue
        try:
            if is_scanned(pdf):
                if not use_ocr:
                    tally["scanned_skipped"] += 1
                    scanned_skipped.append(pdf.name)
                    print(f"  [{n:>3}/{len(pdfs)}] SCANNED  {pdf.name}  (needs OCR - skipped)")
                    continue
                body = extract_ocr(pdf, ocr, args.ocr_dpi)
                kind = "ocr"
            else:
                body = extract_digital(pdf)
                kind = "digital"

            if not body:
                raise ValueError("no content extracted")
            header = f"# {pdf.stem}\n\n> source: {pdf.name} | method: {kind}\n\n"
            out_file.write_text(header + body, encoding="utf-8")
            tally[kind] += 1
            print(f"  [{n:>3}/{len(pdfs)}] {kind:<7}  {pdf.name}  ({len(body):,} chars)")
        except Exception as exc:
            tally["fail"] += 1
            failures.append((pdf.name, str(exc)))
            print(f"  [{n:>3}/{len(pdfs)}] FAIL     {pdf.name}  ({exc})")

    print("\n" + "=" * 60)
    print(f"digital: {tally['digital']}   ocr: {tally['ocr']}   skipped(done): {tally['skip']}"
          f"   scanned-skipped: {tally['scanned_skipped']}   failed: {tally['fail']}")
    print(f"Markdown written to: {out_dir.resolve()}")
    if scanned_skipped:
        print(f"\nScanned (install rapidocr-onnxruntime, then re-run to capture these {len(scanned_skipped)}):")
        for name in scanned_skipped:
            print(f"  - {name}")
    if failures:
        print("\nFailed (re-run to retry):")
        for name, detail in failures:
            print(f"  - {name}: {detail}")


if __name__ == "__main__":
    main()
