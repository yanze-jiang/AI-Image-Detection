from __future__ import annotations

import argparse
import csv
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_DIR_NAMES = {"real", "nature", "original", "0_real", "imagenet"}
FAKE_DIR_NAMES = {"fake", "ai", "ai_fake", "generated", "1_fake"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val/test CSV manifests.")
    parser.add_argument("--tiny-genimage-root", type=Path, default=Path("data/tiny-genimage"))
    parser.add_argument("--mnw-root", type=Path, default=Path("data/MNW/AI_Images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    return parser.parse_args()


def iter_images(root: Path):
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def label_from_path(path: Path, split_root: Path) -> str | None:
    parts = {part.lower() for part in path.relative_to(split_root).parts[:-1]}
    if parts & REAL_DIR_NAMES:
        return "real"
    if parts & FAKE_DIR_NAMES:
        return "fake"
    if len(path.relative_to(split_root).parts) > 1:
        return "fake"
    return None


def generator_from_path(path: Path, split_root: Path, label: str) -> str:
    if label == "real":
        return "real"
    relative_parts = path.relative_to(split_root).parts
    return relative_parts[0] if len(relative_parts) > 1 else "unknown"


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split", "generator"])
        writer.writeheader()
        writer.writerows(rows)


def build_split_manifest(dataset_root: Path, split: str) -> list[dict[str, str]]:
    split_root = dataset_root / split
    rows: list[dict[str, str]] = []
    for image_path in iter_images(split_root) or []:
        label = label_from_path(image_path, split_root)
        if label is None:
            continue
        rows.append(
            {
                "path": str(image_path),
                "label": label,
                "split": split,
                "generator": generator_from_path(image_path, split_root, label),
            }
        )
    return rows


def build_mnw_manifest(mnw_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for image_path in iter_images(mnw_root) or []:
        relative_parts = image_path.relative_to(mnw_root).parts
        generator = relative_parts[0] if len(relative_parts) > 1 else "unknown"
        rows.append(
            {
                "path": str(image_path),
                "label": "fake",
                "split": "test",
                "generator": generator,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    train_rows = build_split_manifest(args.tiny_genimage_root, "train")
    val_rows = build_split_manifest(args.tiny_genimage_root, "val")
    test_rows = build_mnw_manifest(args.mnw_root)

    write_rows(args.output_dir / "train.csv", train_rows)
    write_rows(args.output_dir / "val.csv", val_rows)
    write_rows(args.output_dir / "test.csv", test_rows)

    print(f"train: {len(train_rows)} rows")
    print(f"val: {len(val_rows)} rows")
    print(f"test: {len(test_rows)} rows")


if __name__ == "__main__":
    main()
