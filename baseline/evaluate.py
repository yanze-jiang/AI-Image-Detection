from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from common import (
    build_evaluation_entry,
    build_clip_classifier,
    build_loader,
    build_record_base,
    build_mobilenet_v3_small,
    build_resnet18,
    evaluate,
    get_device,
    result_to_metrics,
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
        backbone = "ResNet18"
        frozen_backbone = False
    elif model_type == "mobilenet_v3_small":
        model = build_mobilenet_v3_small(num_classes=2, pretrained=False)
        transform_model_type = "resnet"
        backbone = "MobileNetV3-Small"
        frozen_backbone = False
    elif model_type == "clip":
        model = build_clip_classifier(model_name=checkpoint["clip_model_name"], num_classes=2)
        transform_model_type = "clip"
        backbone = "CLIP ViT-B/32"
        frozen_backbone = True
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
    eval_type = "robustness" if args.split != "test" else "in_domain_test"
    perturbation_type = "none"
    jpeg_quality: int | None = None
    resize_setting: str | None = None
    if args.split.startswith("test_jpeg"):
        perturbation_type = "jpeg"
        jpeg_quality = int(args.split.replace("test_jpeg", ""))
    elif args.split == "test_resize":
        perturbation_type = "resize"
        resize_setting = "需手动填写"

    payload = build_record_base(
        record_type="eval_only",
        stage="baseline",
        tags=["evaluation", model_type, eval_type],
        seed=4210,
        device=device,
        script_path="baseline/evaluate.py",
    )
    payload["experiment_id"] = f"{model_type}_{args.split}_eval"
    payload["run_id"] = "需手动填写"
    payload["data"].update(
        {
            "train_data_root": "需手动填写",
            "val_data_root": "需手动填写",
            "test_data_root": str(data_root),
            "benchmark_data_root": "需手动填写",
            "train_split": "train",
            "val_split": "val",
            "test_split": args.split,
            "bias_control": True,
            "train_size_per_class": "需手动填写",
            "input_resolution": 224,
            "save_format": "jpeg",
            "preprocess_note": "需手动填写",
        }
    )
    payload["model"].update(
        {
            "model_name": model_type,
            "method_family": "baseline",
            "backbone": backbone,
            "pretrained": checkpoint.get("pretrained", True),
            "frozen_backbone": frozen_backbone,
            "forensic_branch": "none",
            "fusion_method": "none",
        }
    )
    payload["training"]["enabled"] = False
    payload["training"]["checkpoint_path"] = str(args.checkpoint)
    payload["training"]["in_domain_test_metrics"] = result_to_metrics(result)
    payload["evaluations"] = [
        build_evaluation_entry(
            checkpoint_path=args.checkpoint,
            eval_type=eval_type,
            data_root=data_root,
            split=args.split,
            perturbation_type=perturbation_type,
            jpeg_quality=jpeg_quality,
            resize_setting=resize_setting,
            result=result,
            num_samples=len(loader.dataset),
            note="自动记录：评估脚本输出",
        )
    ]
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.save_json is not None:
        save_json(args.save_json, payload)


if __name__ == "__main__":
    main()
