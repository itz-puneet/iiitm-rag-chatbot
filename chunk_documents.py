"""
Chunk the extracted college-policy Markdown files for RAG, with rich metadata.

Two stages:
  1. Recursive Character Text Splitting (LangChain's RecursiveCharacterTextSplitter).
     It tries to split on the largest natural boundary first and only descends to
     finer ones when a piece is still too big:
        paragraphs (\\n\\n) -> lines (\\n) -> sentences (. ! ? ; :) -> words -> chars
     so crucial context (a whole paragraph / sentence) is kept together.

  2. Metadata tagging. extract_metadata() attaches to every chunk:
        - source   : the original PDF name (e.g. 'M.Tech-in-CSE-Updated-2025.pdf')
        - audience : who it applies to  (BTech / MTech / MBA / MS / PhD / IPG / General)
        - topic    : what it's about    (Fees / Admissions / Curriculum / Hostel / ...)
        - section  : nearest Markdown heading above the chunk
     ...so you can filter retrieval later, e.g.  audience == "MTech" AND topic == "Fees".

Setup:
    pip install langchain-text-splitters

Usage:
    python chunk_documents.py                                  # ./extracted -> chunks.jsonl
    python chunk_documents.py --in ./extracted --out chunks.jsonl --size 1000 --overlap 150
"""

import argparse
import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------------------------------- #
#  Metadata vocabularies (tune these to your corpus)
# --------------------------------------------------------------------------- #
# Distinctive keywords per audience. A filename match tags the whole document;
# otherwise audience is detected per chunk from the chunk text + its heading
# breadcrumb. Matching is word-boundary regex where a space flexes to any run of
# space/hyphen/underscore (see _compile), so hyphen/underscore filenames like
# 'Master-of-Business-Administration' and 'MS_AI_DS' match the space keywords,
# while the \b anchor blocks substring false positives ('programs aim' != MS).
# Department names are included only where they map to exactly one program
# ('management studies' -> MBA); shared departments like Computer Science (both
# BTech and MTech) are intentionally omitted.
AUDIENCE_KEYWORDS = {
    "BTech": ["b.tech", "btech", "b. tech", "bachelor of technology"],
    "MTech": ["m.tech", "mtech", "m. tech", "master of technology"],
    "MBA":   ["mba", "master of business administration", "business administration",
              "management studies"],
    "MS":    ["master of science", "m.s.", "ms in artificial", "ms ai", "ms (ai"],
    "PhD":   ["ph.d", "phd", "doctor of philosophy", "doctoral"],
    "IPG":   ["integrated postgraduate", "integrated post graduate", "ipg", "dual degree"],
}

# Topic -> keyword list. A chunk's topic is the highest-scoring topic in it
# (filename keywords are weighted higher); falls back to the document's topic.
TOPIC_KEYWORDS = {
    "Fees":            ["fee", "fees", "tuition", "refund", "payment", "caution money", "deposit"],
    "Scholarship":     ["scholarship", "fellowship", "financial assistance", "freeship", "mcm", "stipend"],
    "Admissions":      ["admission", "counselling", "eligibility", "prospectus", "entrance", "seat"],
    "Curriculum":      ["curriculum", "course of study", "syllabus", "credit", "elective", "semester", "scheme", "schema"],
    "Examination":     ["examination", "grading", "cgpa", "sgpa", "evaluation", "re-evaluation", "attendance"],
    "Hostel":          ["hostel", "mess", "accommodation", "warden", "room and basic"],
    "Ordinance/Rules": ["ordinance", "regulation", "statute", "rules", "act ", "bye-law"],
    "AcademicCalendar":["academic calendar", "calendar", "holiday", "vacation", "commencement"],
    "Placement":       ["placement", "recruit", "internship", "training", "career"],
    "Research":        ["research", "consultancy", "sponsored", "thesis", "synopsis", "publication"],
    "Library":         ["library", "book bank", "journal"],
    "Governance":      ["board of governors", "senate", "finance committee", "minutes", "bog", "notification"],
}


def _compile(keyword: str) -> "re.Pattern":
    """Compile a keyword to a boundary-anchored regex where any space flexes to a
    run of space/hyphen/underscore. So 'ms ai' matches 'MS_AI'/'MS-AI'/'ms ai'.
    The leading (?<![a-z0-9]) blocks letter/digit-adjacent substrings ('programs
    aim' != 'ms ai') while still allowing '_'/'-' separators before the keyword
    (which \\b would wrongly reject, since '_' is a regex word character)."""
    esc = re.escape(keyword.lower()).replace("\\ ", r"[-_\s]+").replace(" ", r"[-_\s]+")
    return re.compile(r"(?<![a-z0-9])" + esc)


AUDIENCE_PATTERNS = {a: [_compile(k) for k in kws] for a, kws in AUDIENCE_KEYWORDS.items()}
TOPIC_PATTERNS = {t: [_compile(k) for k in kws] for t, kws in TOPIC_KEYWORDS.items()}


def audiences_in(text: str) -> set[str]:
    """Every audience whose keywords appear in `text` (word-boundary, separator-flexible)."""
    tx = text.lower()
    return {a for a, pats in AUDIENCE_PATTERNS.items() if any(p.search(tx) for p in pats)}


def filename_audience(filename: str) -> list[str]:
    """Audience(s) named in the filename. If present, these are authoritative for
    the whole document (e.g. 'M.Tech-in-CSE-...pdf' -> every chunk is MTech)."""
    return sorted(audiences_in(filename))


def _topic_scores(text: str, filename: str = "") -> dict[str, int]:
    tx, fn = text.lower(), filename.lower()
    scores = {}
    for topic, pats in TOPIC_PATTERNS.items():
        score = sum(len(p.findall(tx)) for p in pats) + 3 * sum(1 for p in pats if p.search(fn))
        if score:
            scores[topic] = score
    return scores


def detect_topic(text: str, filename: str, fallback: str) -> str:
    scores = _topic_scores(text, filename)
    return max(scores, key=scores.get) if scores else fallback


def _clean_title(title: str) -> str:
    """Strip Markdown emphasis/underline markup from a heading for display."""
    return re.sub(r"[*_`#]|</?u>", "", title).strip()


def breadcrumb_at(start_index, headings):
    """The stack of ancestor headings above `start_index`, by Markdown level.
    A chunk deep in 'SEMESTER-III' still inherits its program heading
    '## M. Tech. (Information and Cyber Security)' from higher up."""
    stack = []  # list of (level, title)
    for pos, level, title in headings:
        if pos > start_index:
            break
        while stack and stack[-1][0] >= level:   # a heading closes deeper siblings
            stack.pop()
        stack.append((level, title))
    return [t for _, t in stack]


def extract_metadata(chunk_text, start_index, *, source_pdf, headings,
                     name_audience, doc_topic, chunk_id):
    """Build the metadata dict appended to a single chunk."""
    crumb = breadcrumb_at(start_index, headings)            # ancestor headings
    crumb_text = " ".join(crumb)
    section = _clean_title(crumb[-1]) if crumb else ""
    # Audience: trust the filename if it names a program; otherwise detect from
    # THIS chunk's text + its full heading breadcrumb, so a chunk under a program
    # section gets that program even when its nearest heading is 'SEMESTER-III'.
    if name_audience:
        audience = name_audience
    else:
        found = audiences_in(chunk_text + " " + crumb_text)
        audience = sorted(found) if found else ["General"]
    return {
        "source": source_pdf,                                   # original PDF name
        "audience": audience,                                   # list, for filtering
        "topic": detect_topic(chunk_text + " " + crumb_text, "", doc_topic),
        "section": section,
        "chunk_id": chunk_id,
        "start_index": start_index,
        "n_chars": len(chunk_text),
    }


def load_body(md_path: Path):
    """Return (clean_body_text, source_pdf_name) from an extracted .md file."""
    raw = md_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^> source:\s*(.+?)\s*\|", raw, re.M)
    source_pdf = m.group(1).strip() if m else md_path.stem + ".pdf"
    # Drop the injected "# stem\n> source: ..." header block and page-marker comments.
    body = re.sub(r"\A#\s.*\n+> source:.*\n+", "", raw)
    body = re.sub(r"<!--.*?-->", "", body)
    return body.strip(), source_pdf


def main() -> None:
    ap = argparse.ArgumentParser(description="Recursive chunking + metadata for RAG.")
    ap.add_argument("--in", dest="in_dir", default="./extracted")
    ap.add_argument("--out", dest="out_file", default="./chunks.jsonl")
    ap.add_argument("--size", type=int, default=1000, help="target chunk size (chars)")
    ap.add_argument("--overlap", type=int, default=150, help="chunk overlap (chars)")
    args = ap.parse_args()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.size,
        chunk_overlap=args.overlap,
        add_start_index=True,           # gives each chunk its offset -> section lookup
        keep_separator=True,            # keep sentence punctuation with the sentence
        # Largest -> smallest natural boundary. "।" = Hindi danda (bilingual docs).
        separators=["\n\n", "\n", "। ", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
    )
    heading_re = re.compile(r"^(#{1,6})\s+(.*)$", re.M)

    md_files = sorted(Path(args.in_dir).glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files in {Path(args.in_dir).resolve()}")

    all_chunks = []
    for md_path in md_files:
        body, source_pdf = load_body(md_path)
        if not body:
            continue
        name_audience = filename_audience(source_pdf)   # authoritative if non-empty
        doc_topic = detect_topic(body, source_pdf, "General")
        headings = [(m.start(), len(m.group(1)), m.group(2).strip())
                    for m in heading_re.finditer(body)]

        docs = splitter.create_documents([body])
        for i, doc in enumerate(docs):
            meta = extract_metadata(
                doc.page_content, doc.metadata["start_index"],
                source_pdf=source_pdf, headings=headings,
                name_audience=name_audience, doc_topic=doc_topic, chunk_id=i,
            )
            all_chunks.append({"text": doc.page_content, "metadata": meta})

    with open(args.out_file, "w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ---- summary ----
    from collections import Counter
    aud = Counter(a for c in all_chunks for a in c["metadata"]["audience"])
    top = Counter(c["metadata"]["topic"] for c in all_chunks)
    print(f"{len(md_files)} docs -> {len(all_chunks)} chunks  ->  {args.out_file}")
    print(f"avg {sum(c['metadata']['n_chars'] for c in all_chunks)//max(len(all_chunks),1)} chars/chunk\n")
    print("chunks by audience:", dict(aud.most_common()))
    print("chunks by topic   :", dict(top.most_common()))


if __name__ == "__main__":
    main()
