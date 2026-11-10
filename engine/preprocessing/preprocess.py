"""
Expand Exploit-DB documents using a local llama.cpp server.

Expected project structure:

WPI_CS547/
├── WPI_CS547_CyberDocExpansionSearch-main/
│   └── engine/
│       └── preprocessing/
│           └── preprocess.py
└── WPI_CS547_ExploitDB_Small/
    ├── files_exploits.csv
    └── exploits/

Run:
    python preprocess.py

The original Exploit-DB corpus is never modified.
"""

from __future__ import annotations

import csv
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "WPI_CS547_ExploitDB_Small"
"""
Note that the csv file is used to process documents: only docs in the csv 
file are processed.
"""

CSV_PATH = DATASET_ROOT / "files_exploits.csv"
OUTPUT_ROOT = SCRIPT_DIR / "expanded_exploits"

API_BASE = "http://127.0.0.1:8080/v1"
CHAT_ENDPOINT = f"{API_BASE}/chat/completions"
MODEL_NAME = "lfm2.5-8b-a1b-heretic"

# Generation settings
MAX_SOURCE_CHARS = 28_000
MAX_NEW_TOKENS = 1600
REQUEST_TIMEOUT_SECONDS = 240
RETRIES = 2
TEMPERATURE = 0.1

# Process only three documents during initial testing.
# Change to None when ready to process the entire corpus.
MAX_DOCUMENTS: int | None = None

# Skip files that have already been generated.
RESUME = True

LOG_PATH = SCRIPT_DIR / "preprocess.log"


# ============================================================
# DESCRIPTOR DEFINITIONS
# ============================================================

DESCRIPTOR_KEYS = [
    "technical_document_summary",
    "layman_document_summary",
    "keywords",
    "related_terms",
    "entities_attributes",
    "vulnerability_type",
]

DESCRIPTOR_LABELS = {
    "technical_document_summary": "Technical Document Summary (TDS)",
    "layman_document_summary": "Layman's Document Summary (LDS)",
    "keywords": "Key Words (KW)",
    "related_terms": "Related Terms",
    "entities_attributes": "Entities/Attributes",
    "vulnerability_type": "Vulnerability Type",
}

# These markers are used only in the model response.
# They are replaced with the final document section markers.
MODEL_SECTION_MARKERS = {
    "TDS": "technical_document_summary",
    "LDS": "layman_document_summary",
    "KW": "keywords",
    "RELATED_TERMS": "related_terms",
    "ENTITIES_ATTRIBUTES": "entities_attributes",
    "VULNERABILITY_TYPE": "vulnerability_type",
}


# ============================================================
# MODEL PROMPTS
# ============================================================

SYSTEM_PROMPT = """
You are an information-retrieval analyst specializing in cybersecurity
and software vulnerabilities.

Analyze the supplied Exploit-DB document and its metadata.

Return six useful, human-readable descriptor sections in the exact
plain-text format requested by the user.

IMPORTANT:
- Do NOT return JSON.
- Do NOT return XML.
- Do NOT use Markdown code fences.
- Do NOT include commentary or introductory text.
- Do NOT reproduce the original exploit source.
- Do NOT invent facts.
- Do NOT explain your thought process.
- If information is unavailable, place "Not specified in source" in the appropriate field.
- Every section must contain actual descriptor content.
- Return ONLY the six descriptor sections, in the order requested, with no extra text.
"""


USER_PROMPT_TEMPLATE = """
Analyze this Exploit-DB entry for a document-expansion search index.

Produce the following six sections.

1. TDS (Technical Document Summary)

Write 2-4 sentences explaining:
- The technical mechanism.
- The vulnerability or weakness.
- The affected component or technology, if identifiable.
- What the exploit does.

Use accurate technical terminology. Do not claim successful exploitation
unless the source demonstrates it.

2. LDS (Layman's Document Summary)

Write 2-3 sentences explaining in plain language:
- What the weakness is.
- What the exploit allows an attacker to do.

Assume the reader has basic computer literacy but is not a security
specialist. Avoid unexplained jargon.

3. KW (Key Words)

Provide 5-15 high-value search keywords or phrases, separated by commas.

Include relevant technologies, protocols, vulnerability concepts, and
exploit techniques supported by the source.

4. RELATED_TERMS

Provide useful synonyms, alternate names, and closely related technical
concepts that someone without specialized training might use when searching 
for this issue.

Separate terms with commas.

Do not imply that an unmentioned product is affected.

5. ENTITIES_ATTRIBUTES

Identify relevant entities and attributes, including:
- Software and product names.
- Versions.
- Vendors.
- Operating systems.
- Architectures.
- Programming languages.
- Protocols.
- File paths.
- Functions.
- Ports.
- CVE identifiers.

Do not infer missing versions or CVEs.

6. VULNERABILITY_TYPE

Give the most specific defensible vulnerability classification.

Examples include:
- SQL injection
- Cross-site scripting
- Buffer overflow
- Privilege escalation
- Path traversal
- Command injection
- Authentication bypass
- Denial of service
- Information disclosure

Do not guess. If the source does not support a defensible classification,
say "Not specified in source".

METADATA:
{metadata}

ORIGINAL EXPLOIT-DB DOCUMENT:

<document>
{document}
</document>

============================================================
REQUIRED OUTPUT FORMAT
============================================================

Return plain text only.

Use each delimiter exactly once, in the order shown below.

Put the actual descriptor content immediately after each delimiter.

Do not repeat the delimiters inside the descriptor content.

Do not add an introduction, conclusion, numbering, or other sections.

Do not return the bracketed instructions literally.

Section headings are denoted by <<<section name>>>. Return the heading precisely as detailed below.

<<<TDS>>>
[Technical summary]

<<<LDS>>>
[Layman's summary]

<<<KW>>>
[Comma-separated keywords]

<<<RELATED_TERMS>>>
[Comma-separated related terms]

<<<ENTITIES_ATTRIBUTES>>>
[Entities and attributes]

<<<VULNERABILITY_TYPE>>>
[Vulnerability classification]

Do not include introductory text, explanations, reasoning, or conclusions.
Do not include placeholders or repeat the instructions.
Every section must contain useful content, or "Not specified in source".

Return ONLY the six descriptor sections, in the order requested, with no extra text.
"""


# ============================================================
# LOGGING
# ============================================================

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ============================================================
# FILE PATHS AND READING
# ============================================================

def get_source_path(csv_file_value: str) -> Path:
    """
    Resolve CSV paths whether they include the leading 'exploits/'
    directory or not.
    """

    relative = Path(csv_file_value.replace("\\", "/"))

    candidates = [
        DATASET_ROOT / relative,
        DATASET_ROOT / "exploits" / relative,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find source file: {csv_file_value}\n"
        + "\n".join(str(p) for p in candidates)
    )


def get_output_path(source_path: Path) -> Path:
    relative = source_path.relative_to(DATASET_ROOT)
    return OUTPUT_ROOT / relative


def read_source_text(path: Path) -> str:
    """
    Read text while tolerating common legacy Exploit-DB encodings.
    """

    raw = path.read_bytes()

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="replace")


# ============================================================
# COMMENT FORMATTING
# ============================================================

def detect_comment_style(path: Path) -> tuple[str, str]:
    """
    Return the comment prefix and suffix for a file extension.

    Examples:
        Python: #
        C/C++: //
        SQL: --
        HTML: <!-- -->
        Plain text: no comment syntax
    """

    ext = path.suffix.lower()

    hash_comments = {
        ".py", ".pyw", ".sh", ".bash", ".zsh", ".rb", ".pl",
        ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
        ".r", ".ps1",
    }

    slash_comments = {
        ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp",
        ".java", ".js", ".jsx", ".ts", ".tsx", ".go",
        ".rs", ".swift", ".kt", ".kts", ".scala",
        ".php", ".v", ".sv", ".vh", ".svh", ".m", ".mm",
    }

    sql_comments = {".sql"}
    html_comments = {".html", ".htm", ".xml", ".svg"}
    batch_comments = {".bat", ".cmd"}

    if ext in hash_comments:
        return "# ", ""

    if ext in slash_comments:
        return "// ", ""

    if ext in sql_comments:
        return "-- ", ""

    if ext in html_comments:
        return "<!-- ", " -->"

    if ext in batch_comments:
        return "REM ", ""

    # Plain text and unknown extensions.
    return "", ""


def preserve_python_preamble(
    text: str,
    path: Path,
) -> tuple[str, str]:
    """
    Preserve Python shebangs and encoding declarations at the beginning
    of Python files.

    Returns:
        (preamble, remaining_source)
    """

    if path.suffix.lower() not in {".py", ".pyw"}:
        return "", text

    lines = text.splitlines(keepends=True)
    preamble_lines = []

    # Preserve shebang.
    if lines and lines[0].startswith("#!"):
        preamble_lines.append(lines.pop(0))

    # Preserve a PEP 263 encoding declaration.
    for index in range(min(1, len(lines))):
        if re.search(r"coding[:=]\s*[-\w.]+", lines[index]):
            preamble_lines.append(lines.pop(index))
            break

    return "".join(preamble_lines), "".join(lines)


# ============================================================
# EXPANDED DOCUMENT GENERATION
# ============================================================

def build_expanded_document(
    source_text: str,
    source_path: Path,
    descriptors: dict[str, str],
    published_date: str,
) -> str:

    prefix, suffix = detect_comment_style(source_path)

    preamble, remaining_source = preserve_python_preamble(
        source_text,
        source_path,
    )

    generated_lines = []

    # Publication date comes from CSV metadata, not the LLM.
    if published_date:
        generated_lines.extend([
            f"{prefix}#<-- Publish Date -->#{suffix}",
            f"{prefix}{published_date}{suffix}",
            "",
        ])

    # Add the six generated descriptor sections.
    for key in DESCRIPTOR_KEYS:

        label = DESCRIPTOR_LABELS[key]

        generated_lines.append(
            f"{prefix}#<-- {label} -->#{suffix}"
        )

        value = descriptors[key].strip()

        for line in value.splitlines():
            generated_lines.append(
                f"{prefix}{line.rstrip()}{suffix}"
            )

        generated_lines.append("")

    # Mark the beginning of the original document.
    generated_lines.append(
        f"{prefix}#<-- Original Content -->#{suffix}"
    )

    generated_lines.append("")

    generated_block = (
        "\n".join(generated_lines).rstrip() + "\n\n"
    )

    original = remaining_source.lstrip("\ufeff")

    return preamble + generated_block + original


# ============================================================
# MODEL RESPONSE PARSING
# ============================================================

def parse_plain_text_sections(
    response_text: str,
) -> dict[str, str]:

    text = response_text.strip()

    # Remove empty or populated reasoning tags.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove accidental Markdown code fences.
    text = re.sub(
        r"^\s*```(?:text|markdown)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)

    # Recognize both <<<TDS>>> and <<<TDS>>>.
    # Search anywhere in the response, not just at line beginnings.
    marker_pattern = re.compile(
        r"<{3}\s*"
        r"(TDS|LDS|KW|RELATED_TERMS|"
        r"ENTITIES_ATTRIBUTES|VULNERABILITY_TYPE|METADATA)"
        r"\s*>{2,3}",
        flags=re.IGNORECASE,
    )

    matches = list(marker_pattern.finditer(text))

    if not matches:
        raise ValueError(
            "No recognizable descriptor headings were found."
        )

    descriptors = {}

    for index, match in enumerate(matches):

        marker = match.group(1).upper()

        # Metadata is not a descriptor; use it as a stopping point.
        if marker == "METADATA":
            break

        key = MODEL_SECTION_MARKERS[marker]

        if key in descriptors:
            raise ValueError(
                f"Duplicate descriptor heading: {key}"
            )

        start = match.end()

        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        value = text[start:end].strip()

        # Remove accidental Markdown formatting.
        value = re.sub(
            r"^\s*[-*#>]+\s*",
            "",
            value,
            flags=re.MULTILINE,
        ).strip()

        if not value:
            raise ValueError(
                f"Descriptor section '{key}' is empty."
            )

        descriptors[key] = value

    missing = set(DESCRIPTOR_KEYS) - set(descriptors)

    if missing:
        raise ValueError(
            "Missing descriptor sections: "
            + ", ".join(sorted(missing))
        )

    return descriptors



def get_message_text(message: dict[str, Any]) -> str:
    """Extract only the model's actual response, never its reasoning."""

    content = message.get("content")

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        parts = [
            item["text"]
            for item in content
            if isinstance(item, dict)
            and isinstance(item.get("text"), str)
        ]

        combined = "\n".join(parts).strip()

        if combined:
            return combined

    raise ValueError(
        "The model returned no usable final response. "
        f"Available fields: {list(message.keys())}"
    )



# ============================================================
# LLM GENERATION
# ============================================================

def make_user_prompt(
    document: str,
    metadata: dict[str, str],
) -> str:

    metadata_text = "\n".join(
        f"- {key}: {value}"
        for key, value in metadata.items()
        if value
    )

    if not metadata_text:
        metadata_text = "No metadata supplied."

    return USER_PROMPT_TEMPLATE.format(
        metadata=metadata_text,
        document=document,
    )


def generate_descriptors(
    document: str,
    metadata: dict[str, str],
    session: requests.Session,
) -> tuple[dict[str, str], bool]:

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": make_user_prompt(document, metadata),
        },
    ]

    # Create a unique output directory for this document and run.
    document_id = metadata.get("id", "unknown_id")
    safe_document_id = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        document_id,
    )

    run_id = time.strftime("%Y%m%d_%H%M%S")

    failure_dir = (
        SCRIPT_DIR
        / "failed_generations"
        / safe_document_id
        / run_id
    )

    last_error = None

    for attempt in range(RETRIES + 1):

        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": TEMPERATURE,
            "top_p": 0.9,
            "max_tokens": MAX_NEW_TOKENS,
            "stream": False,
        }

        result = None
        answer = ""

        try:

            response = session.post(
                CHAT_ENDPOINT,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            response.raise_for_status()

            result = response.json()

            message = result["choices"][0]["message"]

            answer = get_message_text(message)

            # Parse and return successfully generated descriptors.
            return parse_plain_text_sections(answer), False

        except (
            requests.RequestException,
            KeyError,
            IndexError,
            ValueError,
        ) as exc:

            last_error = exc

            logging.warning(
                "LLM attempt %d/%d failed for document %s: %s",
                attempt + 1,
                RETRIES + 1,
                document_id,
                exc,
            )

            # Create the failure directory only when needed.
            failure_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Save the raw model response.
            response_path = (
                failure_dir
                / f"attempt_{attempt + 1}_response.txt"
            )

            response_path.write_text(
                answer if answer else "<NO MODEL TEXT RECEIVED>",
                encoding="utf-8",
            )

            # Save the associated error.
            error_path = (
                failure_dir
                / f"attempt_{attempt + 1}_error.txt"
            )

            error_path.write_text(
                f"Document ID: {document_id}\n"
                f"Attempt: {attempt + 1}/{RETRIES + 1}\n"
                f"Error type: {type(exc).__name__}\n"
                f"Error message: {exc}\n",
                encoding="utf-8",
            )

            # Save the complete API response when available.
            if result is not None:

                api_path = (
                    failure_dir
                    / f"attempt_{attempt + 1}_api_response.txt"
                )

                api_path.write_text(
                    repr(result),
                    encoding="utf-8",
                )

            print(
                f"\n[Generation failure] Document: {document_id}"
                f"\n  Attempt: {attempt + 1}/{RETRIES + 1}"
                f"\n  Saved response: {response_path}"
                f"\n  Saved error: {error_path}\n"
            )

            # Include the failed assistant response in retry history.
            if answer:
                messages.append({
                    "role": "assistant",
                    "content": answer,
                })

            messages.append({
                "role": "user",
                "content": (
                    "Your previous response did not follow the required "
                    "plain-text format. Do NOT output JSON. Return all six "
                    "sections using these exact markers: "
                    "<<<TDS>>>, <<<LDS>>>, <<<KW>>>, "
                    "<<<RELATED_TERMS>>>, <<<ENTITIES_ATTRIBUTES>>>, "
                    "<<<VULNERABILITY_TYPE>>>. "
                    "Put substantive text under every marker. "
                    "Do not use placeholders."
                ),
            })

    logging.warning(
        "All LLM attempts failed for document %s. "
        "Using fallback descriptors.",
        document_id,
    )

    return {
        key: "Failed to generate"
        for key in DESCRIPTOR_KEYS
    }, True



# ============================================================
# CSV PROCESSING
# ============================================================

def load_rows() -> list[dict[str, str]]:

    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}"
        )

    with CSV_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        if "file" not in reader.fieldnames:
            raise ValueError(
                "CSV is missing the expected 'file' column. "
                f"Found columns: {reader.fieldnames}"
            )

        return [
            dict(row)
            for row in reader
            if row.get("file")
        ]


def get_metadata(
    row: dict[str, str],
) -> dict[str, str]:

    wanted = [
        "id",
        "description",
        "date_published",
        "author",
        "type",
        "platform",
        "port",
        "verified",
        "codes",
        "tags",
        "aliases",
    ]

    return {
        key: str(row.get(key, "") or "").strip()
        for key in wanted
        if str(row.get(key, "") or "").strip()
    }


# ============================================================
# SERVER CONNECTION
# ============================================================

def check_server(session: requests.Session) -> None:

    try:

        response = session.get(
            f"{API_BASE}/models",
            timeout=10,
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        raise RuntimeError(
            "Could not connect to llama.cpp at "
            "http://127.0.0.1:8080. "
            "Start llama-server first and leave its terminal open."
        ) from exc


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_one(
    row: dict[str, str],
    session: requests.Session,
) -> tuple[bool, bool, float]:

    csv_file = str(row.get("file", "")).strip()

    source_path = get_source_path(csv_file)

    output_path = get_output_path(source_path)

    # Skip files already generated.
    if RESUME and output_path.is_file():
        return True, False, 0.0

    source_text = read_source_text(source_path)

    if not source_text.strip():
        raise ValueError("Source file is empty.")

    # Truncate long source documents for model input.
    model_input = source_text

    if len(model_input) > MAX_SOURCE_CHARS:

        model_input = (
            model_input[:MAX_SOURCE_CHARS]
            + "\n\n[Document truncated for descriptor generation.]"
        )

    started = time.perf_counter()

    descriptors, llm_failed = generate_descriptors(
        model_input,
        get_metadata(row),
        session,
    )

    elapsed = time.perf_counter() - started

    expanded = build_expanded_document(
        source_text=source_text,
        source_path=source_path,
        descriptors=descriptors,
        published_date=str(
            row.get("date_published", "") or ""
        ).strip(),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        expanded,
        encoding="utf-8",
        newline="",
    )

    return False, llm_failed, elapsed



# ============================================================
# MAIN
# ============================================================

def main() -> None:

    configure_logging()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Dataset root: %s", DATASET_ROOT)
    logging.info("CSV: %s", CSV_PATH)
    logging.info("Output: %s", OUTPUT_ROOT)

    rows = load_rows()

    if MAX_DOCUMENTS is not None:
        rows = rows[:MAX_DOCUMENTS]

    logging.info(
        "Documents selected: %d",
        len(rows),
    )

    with requests.Session() as session:

        check_server(session)

        processed = 0
        skipped = 0
        failed = 0

        inference_seconds = []

        for row in tqdm(
            rows,
            desc="Expanding Exploit-DB documents",
        ):

            try:

                was_skipped, llm_failed, elapsed = process_one(
                    row,
                    session,
                )

                if was_skipped:

                    skipped += 1

                else:

                    processed += 1
                    inference_seconds.append(elapsed)

                    logging.info(
                        "Processed %s in %.2f seconds",
                        row.get("file", "<unknown>"),
                        elapsed,
                    )

                    if llm_failed:
                        failed += 1

            except Exception:

                failed += 1

                logging.exception(
                    "Failed document: %s",
                    row.get("file", "<unknown>"),
                )

        logging.info("Finished.")

        logging.info(
            "Processed: %d | Already existed: %d | Failed: %d",
            processed,
            skipped,
            failed,
        )

        if inference_seconds:

            average = (
                sum(inference_seconds)
                / len(inference_seconds)
            )

            logging.info(
                "Average LLM time/document: %.2f seconds",
                average,
            )

            logging.info(
                "Estimated time for 50,000 documents: %.1f hours",
                average * 50_000 / 3600,
            )



if __name__ == "__main__":
    main()
