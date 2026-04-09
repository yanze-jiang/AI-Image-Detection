from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from common import (
    EvalResult,
    build_evaluation_entry,
    build_loader,
    build_record_base,
    build_resnet18,
    evaluate,
    get_device,
    result_to_metrics,
    save_checkpoint,
    save_json,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet18 baseline on CIFAKE.")
    parser.add_argument("--data-root", type=Path, default=None, help="Processed CIFAKE root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=4210)
    parser.add_argument(
        "--small-train-count",
        type=int,
        default=None,
        help="Use a smaller train subset per class, for example 10000.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained ResNet18 weights.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience on validation AUC.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root or Path("data/processed")
    run_name = (
        f"small_{args.small_train_count}" if args.small_train_count is not None else "full"
    )
    output_dir = args.output_dir or Path("baseline/outputs/resnet18") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Data root: {data_root}")

    train_loader, class_to_idx = build_loader(
        data_root=data_root,
        split="train",
        model_type="resnet",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_subset_per_class=args.small_train_count,
        seed=args.seed,
    )
    val_loader, _ = build_loader(
        data_root=data_root,
        split="val",
        model_type="resnet",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader, _ = build_loader(
        data_root=data_root,
        split="test",
        model_type="resnet",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    positive_index = class_to_idx["fake"]
    model = build_resnet18(num_classes=2, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best_val_auc = -1.0
    best_epoch = -1
    best_val_result: EvalResult | None = None
    last_train_loss: float | None = None
    wait = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            progress_desc=f"train epoch {epoch}",
        )
        val_result = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            positive_index=positive_index,
            progress_desc=f"val epoch {epoch}",
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_result.loss,
                "val_accuracy": val_result.accuracy,
                "val_auc": val_result.auc,
                "val_f1": val_result.f1,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_auc={val_result.auc:.4f} val_acc={val_result.accuracy:.4f}"
        )
        last_train_loss = train_loss

        if val_result.auc > best_val_auc:
            best_val_auc = val_result.auc
            best_epoch = epoch
            best_val_result = val_result
            wait = 0
            save_checkpoint(
                checkpoint_path,
                model=model,
                payload={
                    "model_type": "resnet18",
                    "class_to_idx": class_to_idx,
                    "small_train_count": args.small_train_count,
                    "pretrained": not args.no_pretrained,
                },
            )
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        positive_index=positive_index,
        progress_desc="test",
    )

    summary = build_record_base(
        record_type="train_and_eval",
        stage="baseline",
        tags=["baseline", "resnet18"],
        seed=args.seed,
        device=device,
        script_path="baseline/train_resnet18.py",
    )
    summary["experiment_id"] = f"resnet18_{run_name}_train_and_eval"
    summary["run_id"] = f"resnet18_{run_name}"
    summary["data"].update(
        {
            "train_data_root": str(data_root),
            "val_data_root": str(data_root),
            "test_data_root": str(data_root),
            "benchmark_data_root": "需手动填写",
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "bias_control": True,
            "train_size_per_class": (
                str(args.small_train_count)
                if args.small_train_count is not None
                else "full"
            ),
            "input_resolution": 224,
            "save_format": "jpeg",
            "preprocess_note": "需手动填写",
        }
    )
    summary["model"].update(
        {
            "model_name": "resnet18",
            "method_family": "baseline",
            "backbone": "ResNet18",
            "pretrained": not args.no_pretrained,
            "frozen_backbone": False,
            "forensic_branch": "none",
            "fusion_method": "none",
        }
    )
    summary["training"].update(
        {
            "enabled": True,
            "output_dir": str(output_dir),
            "checkpoint_path": str(checkpoint_path),
            "loss_fn": "CrossEntropyLoss",
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "best_epoch": best_epoch,
            "train_loss": last_train_loss,
            "val_metrics": result_to_metrics(best_val_result),
            "in_domain_test_metrics": result_to_metrics(test_result),
        }
    )
    summary["evaluations"] = [
        build_evaluation_entry(
            checkpoint_path=checkpoint_path,
            eval_type="in_domain_test",
            data_root=data_root,
            split="test",
            perturbation_type="none",
            result=test_result,
            num_samples=len(test_loader.dataset),
            note="自动记录：标准测试集结果",
        )
    ]
    summary["summary"]["main_finding"] = "需手动填写"
    summary["summary"]["failure_mode"] = "需手动填写"
    summary["summary"]["next_action"] = "需手动填写"
    summary["summary"]["report_sentence"] = "需手动填写"
    summary["history"] = history
    save_json(output_dir / "summary.json", summary)
    print("Final test metrics:")
    print(
        f"acc={test_result.accuracy:.4f} auc={test_result.auc:.4f} "
        f"f1={test_result.f1:.4f}"
    )


if __name__ == "__main__":
    main()
