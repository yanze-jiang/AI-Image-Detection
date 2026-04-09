from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from common import (
    build_loader,
    build_mobilenet_v3_small,
    evaluate,
    get_device,
    get_project_root,
    save_checkpoint,
    save_json,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MobileNetV3-Small baseline on the processed Hemg dataset."
    )
    parser.add_argument(
        "--data-root", type=Path, default=None, help="Processed Hemg root."
    )
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
        help="Disable ImageNet pretrained MobileNetV3-Small weights.",
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
    project_root = get_project_root()
    data_root = args.data_root or (project_root / "data" / "hemg_processed")
    run_name = (
        f"small_{args.small_train_count}" if args.small_train_count is not None else "full"
    )
    output_dir = args.output_dir or (
        project_root / "baseline" / "outputs" / "mobilenet_v3_small" / run_name
    )
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
    model = build_mobilenet_v3_small(
        num_classes=2, pretrained=not args.no_pretrained
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best_val_auc = -1.0
    best_epoch = -1
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

        if val_result.auc > best_val_auc:
            best_val_auc = val_result.auc
            best_epoch = epoch
            wait = 0
            save_checkpoint(
                checkpoint_path,
                model=model,
                payload={
                    "model_type": "mobilenet_v3_small",
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

    summary = {
        "model_type": "mobilenet_v3_small",
        "device": str(device),
        "data_root": str(data_root),
        "small_train_count": args.small_train_count,
        "pretrained": not args.no_pretrained,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_loss": test_result.loss,
        "test_accuracy": test_result.accuracy,
        "test_auc": test_result.auc,
        "test_f1": test_result.f1,
        "history": history,
    }
    save_json(output_dir / "summary.json", summary)
    print("Final test metrics:")
    print(
        f"acc={test_result.accuracy:.4f} auc={test_result.auc:.4f} "
        f"f1={test_result.f1:.4f}"
    )


if __name__ == "__main__":
    main()
