"""Download timestamp-preserving training data: python -m scripts.download_mind."""

import shutil
import urllib.request
import zipfile

from src import config

TRAIN_URL = (
    "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/"
    "MINDsmall_train.zip?download=true"
)


def main() -> None:
    config.TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = config.TRAIN_DIR.parent / "MINDsmall_train.zip"
    if not archive_path.exists():
        partial = archive_path.with_suffix(".zip.part")
        with urllib.request.urlopen(TRAIN_URL, timeout=120) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output)
        partial.replace(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        # Extract only known members; never use paths supplied by the archive.
        for name in ("news.tsv", "behaviors.tsv"):
            target = config.TRAIN_DIR / name
            if not target.exists():
                with archive.open(name) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    print(f"Training data ready: {config.TRAIN_DIR}")


if __name__ == "__main__":
    main()
