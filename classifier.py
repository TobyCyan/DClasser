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

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Keyword taxonomy
# ---------------------------------------------------------------------------

KEYWORDS_DESCRIPTION = """\
ai: AI tools in software engineering
backend: Backend development, cloud services, building servers
code: Coding skills, programming best practices, maintainability
cs2030: CS2030 variants (e.g. CS2030S)
database: Databases, SQL, data management
debugging: Debugging/troubleshooting code
desktop: Desktop apps, GUI, frontend, fullstack
design: Software design, design patterns, product/feature design
documentation: Writing documentation
docker: Containerization with Docker
devops: CI/CD, deployment, DevOps practices
dsa: Data structures and algorithms, CS2040 variants
frameworks: Software frameworks (React, Angular, Django, Flask, Spring)
git: Version control systems
groupwork: Teamwork, collaboration, group projects
ide: IDEs (VSCode, IntelliJ, Eclipse)
industry: Real-world SE practices
internships: Resume/portfolio projects, job preparation
interest: Interest in software engineering
java: Java programming language
large: Explicit mention of large-scale software
network: Networking concepts
orbital: Orbital-style projects
oop: Object-oriented programming
performance: Performance, efficiency, scalability
security: Secure coding practices
soft: Soft skills (communication, time management)
se: General software engineering
testing: Software testing, unit tests, TDD
web: Web development explicitly mentioned
workflow: SDLC, Agile, project workflows
ui: User interface / UX
uml: UML diagrams
unsure: Student unsure of expectations"""

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


# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------

def classify_batch(
    client: OpenAI,
    indexed_texts: list[tuple[int, str]],
    model: str,
) -> dict[int, str]:
    """
    Classify multiple texts in a single API call.
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

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0,
        max_tokens=max(120, 40 * len(indexed_texts)),
    )

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
    for batch in batches:
        # Resolve empty rows and cache hits immediately; collect uncached texts.
        to_classify: list[tuple[int, str]] = []  # (batch-local index, text)

        for j, row in enumerate(batch):
            text = (row.get("Expectations", "") or "").strip()
            if not text:
                row["Keywords"] = ""
            else:
                cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if cache_key in cache:
                    row["Keywords"] = cache[cache_key]
                else:
                    to_classify.append((j, text))

        # One API call covers all uncached rows in the batch.
        if to_classify:
            try:
                result_map = classify_batch(client, to_classify, model)
                for j, text in to_classify:
                    keywords = result_map.get(j, "pending")
                    batch[j]["Keywords"] = keywords
                    cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    cache[cache_key] = keywords
            except Exception as exc:  # noqa: BLE001
                print(f"  API error: {exc}. Marking batch rows as 'pending'.")
                for j, _text in to_classify:
                    batch[j]["Keywords"] = "pending"

        # Print progress for every row in the batch.
        for row in batch:
            processed += 1
            text = (row.get("Expectations", "") or "").strip()
            keywords = row.get("Keywords", "")
            if not text:
                print(f"[{processed:>{pad}}/{total}] (empty) → ''")
            else:
                preview = text[:70].replace("\n", " ")
                print(f"[{processed:>{pad}}/{total}] {preview!r} → {keywords!r}")

        # Persist cache after each batch so progress survives interruptions.
        save_cache(cache, cache_file)

        if delay > 0 and processed < total:
            time.sleep(delay)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Output written to: {output_file}")
    print(f"Cache saved to:          {cache_file}")


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
        default=None,
        help="Path to the output CSV file (default: overwrites the input file)",
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

    output_file = args.output or args.path
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
