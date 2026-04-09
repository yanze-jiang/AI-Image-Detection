from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


SEED = 4210
SOURCE_ROOT = Path("data/Hemg")
OUTPUT_ROOT = Path("data/hemg_processed")
LABEL_MAP = {
    "AiArtData": "fake",
    "RealArt": "real",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the local Hemg dataset into train/val/test with an 8/1/1 ratio."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=SOURCE_ROOT,
        help="Path to the local Hemg dataset saved by datasets.save_to_disk().",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Path to save the split ImageFolder dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for deterministic splitting.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output root before rebuilding.",
    )
    return parser.parse_args()


def resolve_project_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (project_root / path).resolve()


def load_train_split(source_root: Path) -> Dataset:
    dataset = load_from_disk(str(source_root))
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError(f"No train split found in: {source_root}")
        return dataset["train"]
    if isinstance(dataset, Dataset):
        return dataset
    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def split_dataset(dataset: Dataset, seed: int) -> dict[str, Dataset]:
    test_split = dataset.train_test_split(
        test_size=0.1,
        seed=seed,
        stratify_by_column="label",
    )
    train_val = test_split["train"]
    test_set = test_split["test"]
    val_split = train_val.train_test_split(
        test_size=1 / 9,
        seed=seed,
        stratify_by_column="label",
    )
    return {
        "train": val_split["train"],
        "val": val_split["test"],
        "test": test_set,
    }


def normalize_label(raw_label: str) -> str:
    try:
        return LABEL_MAP[raw_label]
    except KeyError as exc:
        raise ValueError(f"Unexpected label name: {raw_label}") from exc


def export_split(
    dataset: Dataset,
    split_name: str,
    output_root: Path,
    label_names: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counters = {"fake": 0, "real": 0}

    for example in dataset:
        label_name = label_names[int(example["label"])]
        label = normalize_label(label_name)
        image = example["image"].convert("RGB")
        index = counters[label]
        counters[label] += 1

        target = output_root / split_name / label / f"{label}_{index:06d}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format="JPEG", quality=95, optimize=True)

        rows.append(
            {
                "split": split_name,
                "label": label,
                "source_label": label_name,
                "path": str(target),
            }
        )

    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "label", "source_label", "path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    source_root = resolve_project_path(project_root, args.source_root)
    output_root = resolve_project_path(project_root, args.output_root)

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = load_train_split(source_root)
    label_feature = dataset.features["label"]
    label_names = list(label_feature.names)
    split_map = split_dataset(dataset, seed=args.seed)

    all_rows: dict[str, list[dict[str, str]]] = {}
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "ratio": {"train": 0.8, "val": 0.1, "test": 0.1},
        "label_names": label_names,
        "counts": {},
    }

    for split_name, split_dataset_obj in split_map.items():
        rows = export_split(
            dataset=split_dataset_obj,
            split_name=split_name,
            output_root=output_root,
            label_names=label_names,
        )
        all_rows[split_name] = rows
        summary["counts"][split_name] = {
            "total": len(rows),
            "fake": sum(1 for row in rows if row["label"] == "fake"),
            "real": sum(1 for row in rows if row["label"] == "real"),
        }

    manifest_root = output_root / "manifests"
    for split_name, rows in all_rows.items():
        write_manifest(manifest_root / f"{split_name}.csv", rows)

    with (output_root / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Finished splitting Hemg dataset.")
    print(f"Source root: {source_root}")
    print(f"Output root: {output_root}")
    for split_name, counts in summary["counts"].items():
        print(
            f"{split_name}: total={counts['total']}, "
            f"fake={counts['fake']}, real={counts['real']}"
        )


if __name__ == "__main__":
    main()
