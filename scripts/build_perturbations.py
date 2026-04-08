from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PERTURBATIONS = ("jpeg95", "jpeg85", "jpeg75", "resize")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create JPEG and resize perturbations for processed test splits."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Processed data root, for example data/processed.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Perturbed data output root, for example data/perturbed.",
    )
    return parser.parse_args()


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def apply_perturbation(image: Image.Image, name: str) -> Image.Image:
    rgb = image.convert("RGB")
    if name == "resize":
        down = rgb.resize((128, 128), Image.Resampling.BICUBIC)
        return down.resize((224, 224), Image.Resampling.BICUBIC)
    return rgb


def jpeg_quality_from_name(name: str) -> int:
    if name == "jpeg95":
        return 95
    if name == "jpeg85":
        return 85
    if name == "jpeg75":
        return 75
    raise ValueError(f"Unknown JPEG perturbation: {name}")


def save_with_perturbation(source: Path, target: Path, perturbation: str) -> bool:
    try:
        with Image.open(source) as image:
            processed = apply_perturbation(image, perturbation)
            target.parent.mkdir(parents=True, exist_ok=True)
            if perturbation == "resize":
                processed.save(target, format="JPEG", quality=95, optimize=True)
            else:
                processed.save(
                    target,
                    format="JPEG",
                    quality=jpeg_quality_from_name(perturbation),
                    optimize=True,
                )
        return True
    except (OSError, UnidentifiedImageError):
        return False


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    split_roots = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and path.name.startswith("test")
    )
    if not split_roots:
        print(f"No test-like splits found under: {source_root}")
        return

    total_written = 0
    for split_root in split_roots:
        split_name = split_root.name
        images = list(iter_images(split_root))
        if not images:
            print(f"Skip empty split: {split_root}")
            continue

        for perturbation in PERTURBATIONS:
            target_split_root = output_root / f"{split_name}_{perturbation}"
            written = 0
            for source in images:
                relative = source.relative_to(split_root)
                target = (target_split_root / relative).with_suffix(".jpg")
                if save_with_perturbation(source, target, perturbation):
                    written += 1
            total_written += written
            print(f"{split_name}_{perturbation}: wrote {written} images")

    print(f"Finished building perturbations. Total files written: {total_written}")


if __name__ == "__main__":
    main()
