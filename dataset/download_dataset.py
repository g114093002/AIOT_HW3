"""Small downloader for the canonical SMS Spam dataset.

Usage:
  python download_dataset.py --out sms_spam_no_header.csv

This script downloads the dataset from the canonical PacktPublishing GitHub
raw URL and writes it to the dataset directory. It avoids adding the full
dataset to the repo by default; use this only if you want a local copy.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from urllib.request import urlopen

URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-"
    "Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)


def download(url: str, out: pathlib.Path) -> None:
    print(f"Downloading dataset from {url} -> {out}")
    resp = urlopen(url)
    # Read as bytes and write to file
    data = resp.read()
    out.write_bytes(data)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="sms_spam_no_header.csv")
    args = parser.parse_args(argv)

    out_path = pathlib.Path(__file__).resolve().parent / args.out
    try:
        download(URL, out_path)
    except Exception as e:
        print("Failed to download dataset:", e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
