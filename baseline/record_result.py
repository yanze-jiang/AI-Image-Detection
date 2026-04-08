from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a result row to results CSV.")
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--subset-version", type=str, default="cifake_v1")
    parser.add_argument("--bias-control", type=str, default="yes")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--train-size", type=str, required=True)
    parser.add_argument("--test-setting", type=str, required=True)
    parser.add_argument("--perturbation", type=str, default="none")
    parser.add_argument("--accuracy", type=float, required=True)
    parser.add_argument("--auc", type=float, required=True)
    parser.add_argument("--f1", type=float, required=True)
    parser.add_argument("--note", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = [
        args.subset_version,
        args.bias_control,
        args.model_name,
        args.train_size,
        args.test_setting,
        args.perturbation,
        f"{args.accuracy:.6f}",
        f"{args.auc:.6f}",
        f"{args.f1:.6f}",
        args.note,
    ]
    with args.csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(f"Appended result to {args.csv_path}")


if __name__ == "__main__":
    main()
