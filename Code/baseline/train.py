from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from common import (
    EvalResult,
    build_loader,
    build_model,
    evaluate,
    get_device,
    save_checkpoint,
    save_json,
    set_seed,
    train_one_epoch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a unified baseline on Tiny-GenImage."
    )
    parser.add_argument(
        "--model",
        choices=["resnet18", "mobilenet_v3_small", "clip"],
        default="resnet18",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "tiny-genimage")
    parser.add_argument("--train-manifest", type=Path, default=PROJECT_ROOT / "data" / "manifests" / "train.csv")
    parser.add_argument("--val-manifest", type=Path, default=PROJECT_ROOT / "data" / "manifests" / "val.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "baseline" / "outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=4210)
    parser.add_argument("--subset-per-class", type=int, default=None)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--clip-model-name", default="openai/clip-vit-base-patch32")
    return parser.parse_args()


def result_to_dict(result: EvalResult | None) -> dict[str, float | None]:
    if result is None:
        return {"loss": None, "accuracy": None, "auc": None, "f1": None}
    return {
        "loss": result.loss,
        "accuracy": result.accuracy,
        "auc": result.auc,
        "f1": result.f1,
    }


def main() -> None:
    args = parse_args()
    run_name = f"{args.model}_seed{args.seed}"
    if args.subset_per_class is not None:
        run_name += f"_subset{args.subset_per_class}"
    output_dir = args.output_dir / run_name
    checkpoint_path = output_dir / "best.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device(args.device)
    model_type_for_transform = "clip" if args.model == "clip" else "cnn"

    print(f"Using device: {device}")
    print(f"Training data: {args.data_root}")
    print("Label convention: real=0, fake=1")

    train_loader, class_to_idx = build_loader(
        data_root=args.data_root,
        split="train",
        model_type=model_type_for_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        manifest_path=args.train_manifest,
        subset_per_class=args.subset_per_class,
        seed=args.seed,
    )
    val_loader, _ = build_loader(
        data_root=args.data_root,
        split="val",
        model_type=model_type_for_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        manifest_path=args.val_manifest,
    )

    model = build_model(
        model_type=args.model,
        pretrained=not args.no_pretrained,
        clip_model_name=args.clip_model_name,
    ).to(device)

    trainable_parameters = model.classifier.parameters() if args.model == "clip" else model.parameters()
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_auc = -1.0
    best_epoch = -1
    best_val_result: EvalResult | None = None
    wait = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            progress_desc=f"train {epoch}",
        )
        val_result = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            progress_desc=f"val {epoch}",
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
            best_val_result = val_result
            wait = 0
            save_checkpoint(
                checkpoint_path,
                model=model,
                payload={
                    "model_type": args.model,
                    "class_to_idx": class_to_idx,
                    "pretrained": not args.no_pretrained,
                    "clip_model_name": args.clip_model_name,
                    "data_root": str(args.data_root),
                    "train_split": "train",
                    "val_split": "val",
                    "label_convention": "real=0,fake=1",
                },
            )
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    summary = {
        "run_name": run_name,
        "model": args.model,
        "dataset": "tiny-genimage",
        "data_root": str(args.data_root),
        "splits": {"train": "train", "val": "val", "test": "MNW"},
        "label_convention": class_to_idx,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "best_epoch": best_epoch,
        "best_val_metrics": result_to_dict(best_val_result),
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }
    save_json(output_dir / "summary.json", summary)
    print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
