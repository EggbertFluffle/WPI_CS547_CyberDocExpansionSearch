#!/usr/bin/env python3
"""Bootstrap engine/indexing: create the pip venv and fetch the prebuilt index."""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

INDEX_DIR = Path(__file__).resolve().parent
VENV_DIR = INDEX_DIR / ".venv"
REQUIREMENTS = INDEX_DIR / "REQUIREMENTS.txt"

# The singular URL (embedding.npy) currently returns 404; the published
# file is named embeddings.npy.
EMBEDDINGS_URL = "https://eggbert.xyz/files/embeddings.npy"
CHUNKS_URL = "https://eggbert.xyz/files/chunks.tsv"


def venv_python() -> Path:
    """Path to the Python interpreter inside .venv on this platform."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv() -> None:
    python = venv_python()
    if python.exists():
        return
    print("Creating .venv and installing REQUIREMENTS.txt ...")
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        check=True,
    )


def download(url: str, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "indexing-setup/1.0"}
    )
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            last_tenth = -1
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                done += len(chunk)
                if total:
                    tenth = done * 10 // total
                    if tenth != last_tenth:
                        print(f"  {min(tenth * 10, 100):3d}%", flush=True)
                        last_tenth = tenth
        tmp.replace(dest)
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Download failed for {url}: {exc}") from exc
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    print(f"  -> {dest.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the index files even if they already exist.",
    )
    args = parser.parse_args()

    ensure_venv()

    for url, name in (
        (EMBEDDINGS_URL, "embeddings.npy"),
        (CHUNKS_URL, "chunks.tsv"),
    ):
        dest = INDEX_DIR / name
        if dest.exists() and not args.force:
            print(f"Already have {dest.name} (pass --force to re-download)")
            continue
        download(url, dest)

    script = INDEX_DIR / "indexing.py"
    print()
    print("Index is ready. Run a query with:")
    print(f'  "{venv_python()}" "{script}" -q "your query"')


if __name__ == "__main__":
    main()
