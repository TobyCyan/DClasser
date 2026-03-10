"""
DClasser - AI-Powered Keyword Classifier for Student Expectations

Usage:
    python classifier.py -p path/to/expectations.csv [--output out.csv] [--model gpt-4o-mini] [--batch 10] [--delay 0.5]

Reads an 'Expectations' column from the CSV and populates a 'Keywords' column
using the OpenAI chat API. Responses are cached in .cache.json to avoid
re-classifying identical entries on subsequent runs.

Required environment variable:
    OPENAI_API_KEY  — set in a .env file (see .env.example)
"""

import os
import csv
import time
import hashlib
import json
import argparse
from pathlib import Path

from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Keyword taxonomy
# ---------------------------------------------------------------------------

KEYWORDS_DESCRIPTION = """\
ai: AI/ML tools in SE
backend: backend/cloud/servers
code: coding skills/best practices
cs2030: CS2030/CS2030S module
database: databases/SQL/data
debugging: debugging/troubleshooting
desktop: desktop/GUI/frontend/fullstack
design: software design/patterns
documentation: writing docs
docker: Docker/containerisation
devops: CI/CD/deployment/DevOps
dsa: data structures/algorithms/CS2040
frameworks: React/Angular/Django/Flask/Spring
git: version control/Git
groupwork: teamwork/collaboration
ide: IDEs/VSCode/IntelliJ
industry: real-world SE practices
internships: resume/portfolio/job prep
interest: interest in SE
java: Java language
large: large-scale software
network: networking concepts
orbital: Orbital-style projects
oop: object-oriented programming
performance: performance/scalability
security: secure coding
soft: soft skills/communication
se: general software engineering
testing: testing/TDD/unit tests
web: web development
workflow: SDLC/Agile/project workflow
ui: UI/UX design
uml: UML diagrams
unsure: student unsure"""

VALID_KEYWORDS: set[str] = {
    "ai", "backend", "code", "cs2030", "database", "debugging", "desktop",
    "design", "documentation", "docker", "devops", "dsa", "frameworks",
    "git", "groupwork", "ide", "industry", "internships", "interest",
    "java", "large", "network", "orbital", "oop", "performance", "security",
    "soft", "se", "testing", "web", "workflow", "ui", "uml", "unsure",
    "pending",
}

SYSTEM_PROMPT = f"""\
You are an AI classifier for a software engineering course.
Your task is to classify student expectations into predefined keywords.

Rules:
- Return ONLY space-separated keywords chosen from the list below.
- Assign every keyword that is relevant; there is no limit.
- If no keyword matches, return exactly the word: pending
- If the student expectation is empty, return an empty string.
- Do NOT include any explanation, punctuation, or extra text.

Keywords and meanings:
{KEYWORDS_DESCRIPTION}"""


# Sentinel values treated as blank (case-insensitive, after stripping whitespace).
_BLANK_VALUES: frozenset[str] = frozenset({"-", "nil", "none", "null", "n/a", "na"})


def is_blank(text: str) -> bool:
    """Return True if the text is empty or is exactly a recognised null-like sentinel
    (ignoring surrounding whitespace and punctuation)."""
    if not text:
        return True
    return text.strip(".,!?;:'/\" ").lower() in _BLANK_VALUES


# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------

def classify_batch(
    client: OpenAI,
    indexed_texts: list[tuple[int, str]],
    model: str,
    max_retries: int = 5,
) -> dict[int, str]:
    """
    Classify multiple texts in a single API call.
    Retries on rate-limit errors with exponential backoff.
    Raises SystemExit immediately on quota exhaustion.
    indexed_texts: list of (batch-local index, text)
    Returns a dict mapping each index to its keyword string.
    """
    numbered = "\n".join(f'{i}: "{text}"' for i, text in indexed_texts)
    user_message = (
        "Classify each of the following student expectations.\n"
        "Return one result per line in the exact format: <number>: <keywords>\n"
        "Rules:\n"
        "- Use only the keywords defined in your instructions.\n"
        "- Assign all relevant keywords, space-separated.\n"
        "- If no keyword matches, write: pending\n"
        "- Do NOT add explanations or extra text.\n\n"
        f"{numbered}\n\nOutput:"
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ]
            )
            break  # success
        except RateLimitError as exc:
            body = getattr(exc, "body", {}) or {}
            if body.get("code") == "insufficient_quota":
                raise SystemExit(
                    "\nError: OpenAI quota exhausted. Check your billing at "
                    "https://platform.openai.com/account/billing\n"
                    "Progress has been saved to the cache — re-run once quota is restored."
                ) from exc
            if attempt == max_retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  Rate limited. Retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)

    raw = response.choices[0].message.content.strip()
    result_map: dict[int, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        num_part, sep, kw_part = line.partition(":")
        if not sep:
            continue
        try:
            idx = int(num_part.strip())
        except ValueError:
            continue
        kw_part = kw_part.strip().lower()
        tokens = kw_part.split()
        valid_tokens = [t for t in tokens if t in VALID_KEYWORDS]
        if valid_tokens:
            result_map[idx] = " ".join(valid_tokens)
        elif kw_part:
            # The model returned something unrecognised — fall back to pending.
            result_map[idx] = "pending"
        else:
            result_map[idx] = ""

    return result_map


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache(path: str) -> dict:
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------

def process_csv(
    input_file: str,
    output_file: str,
    client: OpenAI,
    model: str,
    delay: float,
    cache_file: str,
    batch_size: int = 1,
) -> None:
    cache = load_cache(cache_file)

    with open(input_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"Error: '{input_file}' appears to be empty.")
        fieldnames: list[str] = list(reader.fieldnames)
        rows = list(reader)

    if "Expectations" not in fieldnames:
        raise SystemExit(
            f"Error: 'Expectations' column not found in {input_file}.\n"
            f"Columns found: {fieldnames}"
        )

    if "Keywords" not in fieldnames:
        fieldnames.append("Keywords")

    total = len(rows)
    pad = len(str(total))
    batches = [rows[i:i + batch_size] for i in range(0, total, batch_size)]

    processed = 0
    n_blank = 0
    n_cached = 0
    n_api = 0
    n_pending = 0
    n_errors = 0
    for batch in batches:
        # Resolve empty rows and cache hits immediately; collect uncached texts.
        to_classify: list[tuple[int, str]] = []  # (batch-local index, text)

        for j, row in enumerate(batch):
            text = (row.get("Expectations", "") or "").strip()
            if is_blank(text):
                row["Keywords"] = ""
                n_blank += 1
            else:
                cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if cache_key in cache:
                    row["Keywords"] = cache[cache_key]
                    n_cached += 1
                else:
                    to_classify.append((j, text))

        # One API call covers all uncached rows in the batch.
        if to_classify:
            n_api += len(to_classify)
            try:
                result_map = classify_batch(client, to_classify, model)
                for j, text in to_classify:
                    keywords = result_map.get(j, "pending")
                    batch[j]["Keywords"] = keywords
                    cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    cache[cache_key] = keywords
                    if keywords == "pending":
                        n_pending += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  API error: {exc}. Marking batch rows as 'pending'.")
                n_errors += len(to_classify)
                for j, _text in to_classify:
                    batch[j]["Keywords"] = "pending"

        # Print progress for every row in the batch.
        for row in batch:
            processed += 1
            text = (row.get("Expectations", "") or "").strip()
            keywords = row.get("Keywords", "")
            if is_blank(text):
                print(f"[{processed:>{pad}}/{total}] (blank) → ''")
            else:
                preview = text[:70].replace("\n", " ")
                print(f"[{processed:>{pad}}/{total}] {preview!r} → {keywords!r}")

        # Persist cache after each batch so progress survives interruptions.
        save_cache(cache, cache_file)

        if delay > 0 and processed < total:
            time.sleep(delay)

    try:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        raise SystemExit(
            f"\nError: Cannot write to '{output_file}' — the file may be open in another program (e.g. Excel).\n"
            "Close the file and re-run. All API results are saved in the cache so no calls will be repeated."
        )

    n_classified = n_api - n_errors - n_pending
    print(f"""
─────────────────────────────
 Summary
─────────────────────────────
 Total rows        : {total}
 Blank / skipped   : {n_blank}
 Served from cache : {n_cached}
 Sent to API       : {n_api}
   ↳ Classified    : {n_classified}
   ↳ Pending       : {n_pending}
   ↳ API errors    : {n_errors}
─────────────────────────────
 Output : {output_file}
 Cache  : {cache_file}
─────────────────────────────""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify student expectations in a CSV using the OpenAI API."
    )
    parser.add_argument(
        "--path", "-p",
        default="expectations.csv",
        help="Relative or absolute path to the input CSV file (default: expectations.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results.csv",
        help="Path to the output CSV file (default: results.csv)",
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls to respect rate limits (default: 0.5)",
    )
    parser.add_argument(
        "--cache",
        default=".cache.json",
        help="Path to the response cache file (default: .cache.json)",
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=1,
        help="Rows to include in each prompt sent to the API (default: 1)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Error: OPENAI_API_KEY is not set.\n"
            "Copy .env.example to .env and add your OpenAI API key."
        )

    if not Path(args.path).exists():
        raise SystemExit(f"Error: Input file '{args.path}' not found.")

    output_file = args.output or "results.csv"
    client = OpenAI(api_key=api_key)

    if args.batch < 1:
        raise SystemExit("Error: --batch must be a positive integer.")

    print(f"Model:  {args.model}")
    print(f"Input:  {args.path}")
    print(f"Output: {output_file}")
    print(f"Batch:  {args.batch} row(s) per API call")
    print(f"Delay:  {args.delay}s between batches\n")

    process_csv(
        input_file=args.path,
        output_file=output_file,
        client=client,
        model=args.model,
        delay=args.delay,
        cache_file=args.cache,
        batch_size=args.batch,
    )


if __name__ == "__main__":
    main()
