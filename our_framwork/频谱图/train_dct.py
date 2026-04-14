"""DCT-based AI-generated image detection.

Reference
---------
Frank et al., "Leveraging Frequency Analysis for Deep Fake Image Recognition",
ICML 2020.  https://proceedings.mlr.press/v119/frank20a/frank20a.pdf

Core idea
---------
GAN and diffusion-based generators leave characteristic artifacts in the
frequency domain due to upsampling operations.  This script converts images
to 2-D DCT coefficients (per channel, type-II, orthonormal), applies
log-scaling, and trains a lightweight CNN classifier to distinguish real
images from AI-generated ones.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.fft import dctn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "baseline"))

from common import (
    EvalResult,
    build_evaluation_entry,
    build_record_base,
    get_device,
    result_to_metrics,
    save_checkpoint,
    save_json,
    set_seed,
)


# ---------------------------------------------------------------------------
# DCT Transform
# ---------------------------------------------------------------------------

class DCTTransform:
    """Image → log-scaled DCT coefficients (Frank et al., ICML 2020).

    Pipeline:
      1. Resize(256) → (optional RandomHorizontalFlip) → CenterCrop(target_size)
      2. Pixel values rescaled to [-1, 1]
      3. Per-channel 2-D DCT (type-II, orthonormal)
      4. Log-scale preserving sign: sign(c) · log(|c| + ε)
    """

    def __init__(self, target_size: int = 224, train: bool = False):
        self.target_size = target_size
        ops: list = [transforms.Resize(256)]
        if train:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops.append(transforms.CenterCrop(target_size))
        self.spatial = transforms.Compose(ops)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = self.spatial(img)
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # H×W×C, [-1, 1]
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        channels = []
        for c in range(arr.shape[2]):
            coeffs = dctn(arr[:, :, c], type=2, norm="ortho")
            channels.append(np.sign(coeffs) * np.log(np.abs(coeffs) + 1e-12))
        return torch.from_numpy(np.stack(channels, axis=0))  # C×H×W


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DCTClassifier(nn.Module):
    """Lightweight CNN operating on DCT coefficients.

    Architecture adapted from Frank et al. (ICML 2020) Appendix B.
    An extra pooling stage is added to accommodate 224×224 input
    (the paper used 128×128).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_dct_loader(
    data_root: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    target_size: int = 224,
) -> tuple[DataLoader, dict[str, int]]:
    train = split == "train"
    dataset = datasets.ImageFolder(
        root=str(data_root / split),
        transform=DCTTransform(target_size=target_size, train=train),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() and not torch.backends.mps.is_available(),
        persistent_workers=num_workers > 0,
    )
    return loader, dataset.class_to_idx


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "train",
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    positive_index: int,
    desc: str = "eval",
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    n = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_labels)
    probs = torch.softmax(logits_t, dim=1)
    preds = torch.argmax(probs, dim=1)
    y_true = labels_t.numpy()
    y_pred = preds.numpy()
    y_score = probs[:, positive_index].numpy()
    y_binary = (y_true == positive_index).astype(np.int32)

    return EvalResult(
        loss=total_loss / max(n, 1),
        accuracy=float(accuracy_score(y_true, y_pred)),
        auc=float(roc_auc_score(y_binary, y_score)),
        f1=float(f1_score(y_binary, (y_pred == positive_index).astype(int), zero_division=0)),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DCT-CNN classifier (Frank et al. ICML 2020) on HEMG dataset.",
    )
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=4210)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--save-every", type=int, default=5,
                   help="Save a periodic checkpoint every N epochs.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    data_root = args.data_root or PROJECT_ROOT / "data" / "hemg_processed"
    output_dir = args.output_dir or script_dir / "output"
    checkpoint_dir = args.checkpoint_dir or script_dir / "参数"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Data root: {data_root}")

    train_loader, class_to_idx = build_dct_loader(
        data_root, "train", args.batch_size, args.num_workers,
    )
    val_loader, _ = build_dct_loader(
        data_root, "val", args.batch_size, args.num_workers,
    )
    test_loader, _ = build_dct_loader(
        data_root, "test", args.batch_size, args.num_workers,
    )

    positive_index = class_to_idx["fake"]
    model = DCTClassifier(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {param_count:,}")

    best_val_auc = -1.0
    best_epoch = -1
    best_val: EvalResult | None = None
    last_train_loss: float | None = None
    wait = 0
    history: list[dict] = []
    ckpt_path = checkpoint_dir / "best_dct.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            desc=f"train epoch {epoch}/{args.epochs}",
        )
        val_result = evaluate_model(
            model, val_loader, criterion, device, positive_index,
            desc=f"val epoch {epoch}/{args.epochs}",
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_result.loss,
            "val_accuracy": val_result.accuracy,
            "val_auc": val_result.auc,
            "val_f1": val_result.f1,
        })
        print(
            f"epoch={epoch}  train_loss={train_loss:.4f}  "
            f"val_auc={val_result.auc:.4f}  val_acc={val_result.accuracy:.4f}"
        )
        last_train_loss = train_loss

        if val_result.auc > best_val_auc:
            best_val_auc = val_result.auc
            best_epoch = epoch
            best_val = val_result
            wait = 0
            save_checkpoint(
                ckpt_path,
                model=model,
                payload={"model_type": "dct_cnn", "class_to_idx": class_to_idx},
            )
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % args.save_every == 0:
            periodic_path = checkpoint_dir / f"epoch_{epoch}.pt"
            save_checkpoint(
                periodic_path,
                model=model,
                payload={
                    "model_type": "dct_cnn",
                    "class_to_idx": class_to_idx,
                    "epoch": epoch,
                    "val_auc": val_result.auc,
                },
            )
            print(f"  -> saved periodic checkpoint: {periodic_path}")

    # ---- Final test evaluation with best checkpoint ----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_result = evaluate_model(
        model, test_loader, criterion, device, positive_index, desc="test",
    )

    # ---- Experiment record (following 实验记录模板.json) ----
    summary = build_record_base(
        record_type="train_and_eval",
        stage="improved",
        tags=["frequency", "dct", "frank2020", "hemg"],
        seed=args.seed,
        device=device,
        script_path="our_framwork/频谱图/train_dct.py",
    )
    summary["experiment_id"] = "dct_cnn_hemg_train_and_eval"
    summary["run_id"] = "dct_cnn_hemg"
    summary["data"].update({
        "train_data_root": str(data_root),
        "val_data_root": str(data_root),
        "test_data_root": str(data_root),
        "benchmark_data_root": "需手动填写",
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
        "bias_control": True,
        "train_size_per_class": "full",
        "input_resolution": 224,
        "save_format": "jpeg",
        "preprocess_note": (
            "Resize(256) + CenterCrop(224), 像素归一化到 [-1,1], "
            "逐通道 2D DCT (type-II, ortho) + log-scale"
        ),
    })
    summary["model"].update({
        "model_name": "dct_cnn",
        "method_family": "frequency_analysis",
        "backbone": "DCTClassifier (4-layer CNN on DCT coefficients)",
        "pretrained": False,
        "frozen_backbone": False,
        "forensic_branch": "dct_spectrum",
        "fusion_method": "none",
    })
    summary["training"].update({
        "enabled": True,
        "output_dir": str(output_dir),
        "checkpoint_path": str(ckpt_path),
        "loss_fn": "CrossEntropyLoss",
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "best_epoch": best_epoch,
        "train_loss": last_train_loss,
        "val_metrics": result_to_metrics(best_val),
        "in_domain_test_metrics": result_to_metrics(test_result),
    })
    summary["evaluations"] = [
        build_evaluation_entry(
            checkpoint_path=ckpt_path,
            eval_type="in_domain_test",
            data_root=data_root,
            split="test",
            perturbation_type="none",
            result=test_result,
            num_samples=len(test_loader.dataset),
            note="自动记录：DCT-CNN 在 HEMG 标准测试集上的结果",
        ),
    ]
    summary["history"] = history
    save_json(output_dir / "summary.json", summary)

    print("\n===== Final test metrics =====")
    print(
        f"  accuracy = {test_result.accuracy:.4f}\n"
        f"  auc      = {test_result.auc:.4f}\n"
        f"  f1       = {test_result.f1:.4f}"
    )


if __name__ == "__main__":
    main()
