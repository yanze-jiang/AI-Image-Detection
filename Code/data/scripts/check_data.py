from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check dataset folders and manifest files.")
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args()


def count_manifest_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    root = args.root
    paths = {
        "tiny_train_dir": root / "data" / "tiny-genimage" / "train",
        "tiny_val_dir": root / "data" / "tiny-genimage" / "val",
        "mnw_dir": root / "data" / "MNW" / "AI_Images",
        "train_manifest": root / "data" / "manifests" / "train.csv",
        "val_manifest": root / "data" / "manifests" / "val.csv",
        "test_manifest": root / "data" / "manifests" / "test.csv",
    }

    for name, path in paths.items():
        status = "OK" if path.exists() else "MISSING"
        print(f"{name}: {status} ({path})")

    print("manifest rows:")
    for split in ("train", "val", "test"):
        manifest_path = root / "data" / "manifests" / f"{split}.csv"
        print(f"  {split}: {count_manifest_rows(manifest_path)}")


if __name__ == "__main__":
    main()
