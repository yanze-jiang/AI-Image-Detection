from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample, preprocess, and split a GenImage subset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the JSON config file.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the processed root before rebuilding.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else (base / path).resolve()


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def ensure_count(paths: list[Path], required: int, name: str) -> None:
    if len(paths) < required:
        raise ValueError(
            f"{name} only has {len(paths)} images, but {required} are required."
        )


def sample_without_overlap(
    rng: random.Random, paths: list[Path], counts: dict[str, int]
) -> dict[str, list[Path]]:
    total_needed = sum(counts.values())
    ensure_count(paths, total_needed, "source directory")
    shuffled = paths[:]
    rng.shuffle(shuffled)

    offset = 0
    sampled: dict[str, list[Path]] = {}
    for split_name, count in counts.items():
        sampled[split_name] = shuffled[offset : offset + count]
        offset += count
    return sampled


def save_image(source: Path, target: Path, image_size: int, jpeg_quality: int) -> bool:
    try:
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            resized = rgb.resize((image_size, image_size), Image.Resampling.BICUBIC)
            target.parent.mkdir(parents=True, exist_ok=True)
            resized.save(
                target,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
            )
        return True
    except (OSError, UnidentifiedImageError):
        return False


def process_split(
    split_name: str,
    label: str,
    generator: str,
    sampled_paths: Iterable[Path],
    processed_root: Path,
    image_size: int,
    jpeg_quality: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    split_dir = processed_root / split_name / label
    if label == "ai":
        split_dir = split_dir / generator

    for index, source in enumerate(sampled_paths):
        stem = f"{generator}_{index:05d}.jpg" if label == "ai" else f"real_{index:05d}.jpg"
        target = split_dir / stem
        ok = save_image(source, target, image_size=image_size, jpeg_quality=jpeg_quality)
        if not ok:
            continue
        rows.append(
            {
                "split": split_name,
                "label": label,
                "generator": generator,
                "source_path": str(source),
                "processed_path": str(target),
            }
        )
    return rows


def write_manifest(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "label", "generator", "source_path", "processed_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(output_path: Path, summary: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    workspace_root = config_path.parents[2]
    config = load_config(config_path)

    raw_root = resolve_path(workspace_root, config["raw_root"])
    processed_root = resolve_path(workspace_root, config["processed_root"])
    processed_root.mkdir(parents=True, exist_ok=True)

    if args.clean and processed_root.exists():
        shutil.rmtree(processed_root)
        processed_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config["seed"])

    seen_generators = config["seen_generators"]
    unseen_generator = config["unseen_generator"]

    train_count = int(config["train_count_per_seen_generator"])
    val_count = int(config["val_count_per_seen_generator"])
    test_seen_count = int(config["test_seen_count_per_seen_generator"])
    test_unseen_count = int(config["test_unseen_count"])
    real_counts = config["real_counts"]

    manifests: dict[str, list[dict[str, str]]] = {
        "train": [],
        "val": [],
        "test_seen": [],
        "test_unseen": [],
    }
    summary = {
        "config": config,
        "actual_counts": {
            "train": {"ai": 0, "real": 0},
            "val": {"ai": 0, "real": 0},
            "test_seen": {"ai": 0, "real": 0},
            "test_unseen": {"ai": 0, "real": 0},
        },
    }

    for generator in seen_generators:
        generator_paths = list_images(raw_root / generator)
        sampled = sample_without_overlap(
            rng,
            generator_paths,
            {
                "train": train_count,
                "val": val_count,
                "test_seen": test_seen_count,
            },
        )
        for split_name, paths in sampled.items():
            rows = process_split(
                split_name=split_name,
                label="ai",
                generator=generator,
                sampled_paths=paths,
                processed_root=processed_root,
                image_size=int(config["image_size"]),
                jpeg_quality=int(config["jpeg_quality"]),
            )
            manifests[split_name].extend(rows)
            summary["actual_counts"][split_name]["ai"] += len(rows)

    unseen_paths = list_images(raw_root / unseen_generator)
    ensure_count(unseen_paths, test_unseen_count, unseen_generator)
    unseen_sampled = rng.sample(unseen_paths, test_unseen_count)
    unseen_rows = process_split(
        split_name="test_unseen",
        label="ai",
        generator=unseen_generator,
        sampled_paths=unseen_sampled,
        processed_root=processed_root,
        image_size=int(config["image_size"]),
        jpeg_quality=int(config["jpeg_quality"]),
    )
    manifests["test_unseen"].extend(unseen_rows)
    summary["actual_counts"]["test_unseen"]["ai"] += len(unseen_rows)

    real_paths = list_images(raw_root / "real")
    real_sampled = sample_without_overlap(
        rng,
        real_paths,
        {
            "train": int(real_counts["train"]),
            "val": int(real_counts["val"]),
            "test_seen": int(real_counts["test_seen"]),
            "test_unseen": int(real_counts["test_unseen"]),
        },
    )
    for split_name, paths in real_sampled.items():
        rows = process_split(
            split_name=split_name,
            label="real",
            generator="real",
            sampled_paths=paths,
            processed_root=processed_root,
            image_size=int(config["image_size"]),
            jpeg_quality=int(config["jpeg_quality"]),
        )
        manifests[split_name].extend(rows)
        summary["actual_counts"][split_name]["real"] += len(rows)

    manifest_root = processed_root / "manifests"
    for split_name, rows in manifests.items():
        write_manifest(manifest_root / f"{split_name}.csv", rows)

    write_summary(processed_root / "split_summary.json", summary)

    print("Finished preparing dataset subset.")
    print(f"Processed root: {processed_root}")
    for split_name, counts in summary["actual_counts"].items():
        print(
            f"{split_name}: ai={counts['ai']}, real={counts['real']}"
        )


if __name__ == "__main__":
    main()
