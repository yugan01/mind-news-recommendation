"""Download and safely extract the RecZoo mirror of MIND-small."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


ARCHIVE_URL = (
    "https://huggingface.co/datasets/reczoo/MIND_small_x1/resolve/main/"
    "MIND_small_x1.zip?download=true"
)


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"Unsafe archive member: {member.filename}")
    archive.extractall(destination)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "cp4285-mind-project/1.0"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    archive_path = args.output_dir / "MIND_small_x1.zip"
    extract_dir = args.output_dir / "MIND_small_x1"
    if not archive_path.exists():
        print(f"Downloading {ARCHIVE_URL}")
        download(ARCHIVE_URL, archive_path)
    if not (extract_dir / "news_corpus.tsv").exists():
        print(f"Extracting {archive_path} -> {extract_dir}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            safe_extract(archive, extract_dir)
    if not args.keep_zip:
        archive_path.unlink(missing_ok=True)
    print("MIND-small is ready.")


if __name__ == "__main__":
    main()

