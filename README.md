# DClasser

AI-powered keyword classifier for student expectations. Reads an `expectations.csv` file, sends each student response to the OpenAI API, and writes space-separated keywords back into a `Keywords` column.

## Requirements

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/account/api-keys) with available credits

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
```

## Input format

The input CSV must contain an `Expectations` column. A `Keywords` column will be added (or overwritten) automatically.

```
Expectations,Keywords
I want to learn React and web development,
Hope to understand OOP and design patterns,
```

## Usage

```bash
# Basic — reads expectations.csv in the current directory
python classifier.py

# Specify input path
python classifier.py -p data/expectations.csv

# Write results to a separate file
python classifier.py -p data/expectations.csv -o data/results.csv

# Send 10 rows per API call (one prompt, one response)
python classifier.py -p data/expectations.csv -b 10

# Full options
python classifier.py -p data/expectations.csv -o results.csv -b 20 -d 1.0 -m gpt-4o
```

## CLI flags

| Flag | Short | Default | Description |
|---|---|---|---|
| `--path` | `-p` | `expectations.csv` | Path to the input CSV |
| `--output` | `-o` | *(same as input)* | Path to the output CSV |
| `--model` | `-m` | `gpt-4o-mini` | OpenAI model to use |
| `--batch` | `-b` | `1` | Rows packed into each API prompt |
| `--delay` | `-d` | `0.5` | Seconds to wait between API calls |
| `--cache` | | `.cache.json` | Path to the response cache file |

## Keywords

| Keyword | Meaning |
|---|---|
| `ai` | AI/ML tools in SE |
| `backend` | Backend/cloud/servers |
| `code` | Coding skills/best practices |
| `cs2030` | CS2030/CS2030S module |
| `database` | Databases/SQL/data |
| `debugging` | Debugging/troubleshooting |
| `desktop` | Desktop/GUI/frontend/fullstack |
| `design` | Software design/patterns |
| `documentation` | Writing docs |
| `docker` | Docker/containerisation |
| `devops` | CI/CD/deployment/DevOps |
| `dsa` | Data structures/algorithms/CS2040 |
| `frameworks` | React/Angular/Django/Flask/Spring |
| `git` | Version control/Git |
| `groupwork` | Teamwork/collaboration |
| `ide` | IDEs/VSCode/IntelliJ |
| `industry` | Real-world SE practices |
| `internships` | Resume/portfolio/job prep |
| `interest` | Interest in SE |
| `java` | Java language |
| `large` | Large-scale software |
| `network` | Networking concepts |
| `orbital` | Orbital-style projects |
| `oop` | Object-oriented programming |
| `performance` | Performance/scalability |
| `security` | Secure coding |
| `soft` | Soft skills/communication |
| `se` | General software engineering |
| `testing` | Testing/TDD/unit tests |
| `web` | Web development |
| `workflow` | SDLC/Agile/project workflow |
| `ui` | UI/UX design |
| `uml` | UML diagrams |
| `unsure` | Student unsure |
| `pending` | No matching keyword found |

## Behaviour

- **Blank inputs** — empty cells and sentinel values (`-`, `nil`, `none`, `null`, `n/a`) are skipped without calling the API; `Keywords` is left empty.
- **Caching** — each response is cached in `.cache.json` by a SHA-256 hash of the input text. Re-running the script on the same file only calls the API for rows not yet in the cache.
- **Batching** — `-b N` packs N rows into a single numbered prompt and parses the numbered response back. Cache hits within a batch are still resolved locally; only uncached rows are sent.
- **Rate limits** — transient 429 errors are retried up to 5 times with exponential backoff. A quota-exhausted error stops the run immediately and points to the billing page; all progress up to that point is saved in the cache.

## Files

```
classifier.py      # Main script
requirements.txt   # Python dependencies
.env.example       # API key template
.env               # Your API key (git-ignored)
.cache.json        # Response cache (git-ignored)
expectations.csv   # Your input file
```
