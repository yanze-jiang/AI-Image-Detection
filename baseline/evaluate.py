from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from common import (
    build_clip_classifier,
    build_loader,
    build_mobilenet_v3_small,
    build_resnet18,
    evaluate,
    get_device,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a baseline checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root, for example data/processed or data/perturbed.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name under data root, for example test or test_jpeg85.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save evaluation metrics as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root or Path("data/processed")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_type = checkpoint["model_type"]

    if model_type == "resnet18":
        model = build_resnet18(num_classes=2, pretrained=False)
        transform_model_type = "resnet"
    elif model_type == "mobilenet_v3_small":
        model = build_mobilenet_v3_small(num_classes=2, pretrained=False)
        transform_model_type = "resnet"
    elif model_type == "clip":
        model = build_clip_classifier(model_name=checkpoint["clip_model_name"], num_classes=2)
        transform_model_type = "clip"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    device = get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    loader, class_to_idx = build_loader(
        data_root=data_root,
        split=args.split,
        model_type=transform_model_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    result = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        positive_index=class_to_idx["fake"],
        progress_desc=f"eval {args.split}",
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(data_root),
        "split": args.split,
        "model_type": model_type,
        "accuracy": result.accuracy,
        "auc": result.auc,
        "f1": result.f1,
        "loss": result.loss,
    }
    print(payload)
    if args.save_json is not None:
        save_json(args.save_json, payload)


if __name__ == "__main__":
    main()
