from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_MAP = {"REAL": "real", "FAKE": "fake"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CIFAKE into processed train/val/test splits."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the processed root before rebuilding.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else (base / path).resolve()


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def ensure_count(paths: list[Path], required: int, name: str) -> None:
    if len(paths) < required:
        raise ValueError(f"{name} only has {len(paths)} images, but {required} are required.")


def save_image(source: Path, target: Path, image_size: int, jpeg_quality: int) -> bool:
    try:
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            resized = rgb.resize((image_size, image_size), Image.Resampling.BICUBIC)
            target.parent.mkdir(parents=True, exist_ok=True)
            resized.save(target, format="JPEG", quality=jpeg_quality, optimize=True)
        return True
    except (OSError, UnidentifiedImageError):
        return False


def write_manifest(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "label", "source_path", "processed_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def process_split(
    split_name: str,
    label: str,
    sources: list[Path],
    processed_root: Path,
    image_size: int,
    jpeg_quality: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    target_dir = processed_root / split_name / label
    for index, source in enumerate(sources):
        target = target_dir / f"{label}_{index:05d}.jpg"
        ok = save_image(source, target, image_size=image_size, jpeg_quality=jpeg_quality)
        if not ok:
            continue
        rows.append(
            {
                "split": split_name,
                "label": label,
                "source_path": str(source),
                "processed_path": str(target),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    workspace_root = config_path.parents[2]
    config = load_config(config_path)
    config["seed"] = int(config.get("seed", 4210))

    raw_root = resolve_path(workspace_root, config["raw_root"])
    processed_root = resolve_path(workspace_root, config["processed_root"])

    if args.clean and processed_root.exists():
        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config["seed"])
    val_count = int(config["val_count_per_class"])
    use_small_train_subset = bool(config["use_small_train_subset"])
    small_train_count = int(config["small_train_count_per_class"])
    image_size = int(config["image_size"])
    jpeg_quality = int(config["jpeg_quality"])

    manifests: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    summary = {
        "config": config,
        "actual_counts": {
            "train": {"real": 0, "fake": 0},
            "val": {"real": 0, "fake": 0},
            "test": {"real": 0, "fake": 0},
        },
    }

    for raw_label, label in CLASS_MAP.items():
        train_paths = list_images(raw_root / "train" / raw_label)
        test_paths = list_images(raw_root / "test" / raw_label)

        required_train = val_count + (small_train_count if use_small_train_subset else 1)
        ensure_count(train_paths, required_train, f"train/{raw_label}")
        rng.shuffle(train_paths)

        val_sources = train_paths[:val_count]
        remaining_train = train_paths[val_count:]
        if use_small_train_subset:
            train_sources = remaining_train[:small_train_count]
        else:
            train_sources = remaining_train

        test_sources = test_paths

        for split_name, sources in (
            ("train", train_sources),
            ("val", val_sources),
            ("test", test_sources),
        ):
            rows = process_split(
                split_name=split_name,
                label=label,
                sources=sources,
                processed_root=processed_root,
                image_size=image_size,
                jpeg_quality=jpeg_quality,
            )
            manifests[split_name].extend(rows)
            summary["actual_counts"][split_name][label] += len(rows)

    manifest_root = processed_root / "manifests"
    for split_name, rows in manifests.items():
        write_manifest(manifest_root / f"{split_name}.csv", rows)

    with (processed_root / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Finished preparing CIFAKE.")
    print(f"Processed root: {processed_root}")
    for split_name, counts in summary["actual_counts"].items():
        print(f"{split_name}: real={counts['real']}, fake={counts['fake']}")


if __name__ == "__main__":
    main()
