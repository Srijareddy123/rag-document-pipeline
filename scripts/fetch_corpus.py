#!/usr/bin/env python3
"""
Download the public corpus used for the load test and the chunking comparison.

Corpus: Simple English Wikipedia, 20231101 dump from the wikimedia/wikipedia
dataset on Hugging Face (about 157 MB, roughly 230K articles, CC BY-SA).

The file goes to CORPUS_DIR (default ~/.cache/rag-document-pipeline), outside
the repo.

    python scripts/fetch_corpus.py
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
    "20231101.simple/train-00000-of-00001.parquet"
)


def corpus_path() -> Path:
    base = Path(os.getenv("CORPUS_DIR", "~/.cache/rag-document-pipeline")).expanduser()
    return base / "simplewiki-20231101.parquet"


def main() -> int:
    dest = corpus_path()
    if dest.exists():
        print(f"Corpus already at {dest}")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading {URL}\n  to {dest}")
    urllib.request.urlretrieve(URL, tmp)
    tmp.rename(dest)
    print(f"Done, {dest.stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
