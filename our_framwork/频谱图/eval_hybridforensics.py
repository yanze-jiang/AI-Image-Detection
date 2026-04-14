"""Evaluate a DCT-CNN checkpoint on the HybridForensics benchmark.

Usage:
    python eval_hybridforensics.py --checkpoint 参数/best_dct.pt
    python eval_hybridforensics.py --checkpoint 参数/best_dct.pt --save-json output/eval_hybridforensics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from scipy.fft import dctn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "baseline"))

from common import (
    EvalResult,
    build_evaluation_entry,
    build_record_base,
    get_device,
    result_to_metrics,
    save_json,
)
from train_dct import DCTClassifier, DCTTransform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class HybridForensicsDCTDataset(Dataset):
    """HybridForensics dataset with DCT transform for DCT-CNN evaluation."""

    def __init__(self, root: Path, target_size: int = 224):
        self.root = root
        self.transform = DCTTransform(target_size=target_size, train=False)
        self.class_to_idx = {"fake": 0, "real": 1}
        self.samples: list[tuple[Path, int]] = []
        self.skipped_files: list[str] = []
        self._collect_samples()

    def _iter_images(self, root: Path):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path

    def _collect_samples(self) -> None:
        real_root = self.root / "Real"
        fake_gan_root = self.root / "Fake_GAN"
        fake_diff_root = self.root / "Fake_Diffusion"

        for path in self._iter_images(real_root):
            if self._is_readable(path):
                self.samples.append((path, self.class_to_idx["real"]))
        for path in self._iter_images(fake_gan_root):
            if self._is_readable(path):
                self.samples.append((path, self.class_to_idx["fake"]))
        for path in self._iter_images(fake_diff_root):
            if self._is_readable(path):
                self.samples.append((path, self.class_to_idx["fake"]))

        if not self.samples:
            raise ValueError(f"No images found under {self.root}")

    def _is_readable(self, path: Path) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except (OSError, UnidentifiedImageError):
            self.skipped_files.append(str(path))
            return False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return tensor, label


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate DCT-CNN on HybridForensics benchmark.",
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=None,
                   help="HybridForensics raw root. Default: data/hybridforensics/raw")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root or PROJECT_ROOT / "data" / "hybridforensics" / "raw"

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DCTClassifier(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {data_root}")

    dataset = HybridForensicsDCTDataset(root=data_root)
    if dataset.skipped_files:
        print(f"Skipped unreadable files: {len(dataset.skipped_files)}")
        for path in dataset.skipped_files[:10]:
            print(f"  - {path}")
    print(f"Total samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not torch.backends.mps.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    result = evaluate_model(
        model, loader, criterion, device,
        positive_index=dataset.class_to_idx["fake"],
        desc="eval hybridforensics",
    )

    payload = build_record_base(
        record_type="eval_only",
        stage="hybridforensics",
        tags=["evaluation", "dct_cnn", "hybridforensics", "cross_dataset"],
        seed=4210,
        device=device,
        script_path="our_framwork/频谱图/eval_hybridforensics.py",
    )
    payload["experiment_id"] = "dct_cnn_hybridforensics_eval"
    payload["run_id"] = "需手动填写"
    payload["data"].update({
        "train_data_root": "data/hemg_processed",
        "val_data_root": "data/hemg_processed",
        "test_data_root": "需手动填写",
        "benchmark_data_root": str(data_root),
        "train_split": "train",
        "val_split": "val",
        "test_split": "hybridforensics",
        "bias_control": True,
        "train_size_per_class": "full",
        "input_resolution": 224,
        "save_format": "jpeg",
        "preprocess_note": (
            "Resize(256) + CenterCrop(224), 像素归一化到 [-1,1], "
            "逐通道 2D DCT (type-II, ortho) + log-scale"
        ),
    })
    payload["model"].update({
        "model_name": "dct_cnn",
        "method_family": "frequency_analysis",
        "backbone": "DCTClassifier (4-layer CNN on DCT coefficients)",
        "pretrained": False,
        "frozen_backbone": False,
        "forensic_branch": "dct_spectrum",
        "fusion_method": "none",
    })
    payload["training"]["enabled"] = False
    payload["training"]["checkpoint_path"] = str(args.checkpoint)
    payload["training"]["in_domain_test_metrics"] = result_to_metrics(result)
    payload["evaluations"] = [
        build_evaluation_entry(
            checkpoint_path=args.checkpoint,
            eval_type="cross_dataset",
            data_root=data_root,
            split="hybridforensics",
            perturbation_type="none",
            result=result,
            num_samples=len(dataset),
            skipped_files=dataset.skipped_files,
            note="自动记录：DCT-CNN 在 HybridForensics 上的跨数据集评估",
        ),
    ]

    print("\n===== HybridForensics Results =====")
    print(
        f"  accuracy = {result.accuracy:.4f}\n"
        f"  auc      = {result.auc:.4f}\n"
        f"  f1       = {result.f1:.4f}\n"
        f"  loss     = {result.loss:.4f}"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.save_json is not None:
        save_json(args.save_json, payload)
        print(f"\nSaved to {args.save_json}")


if __name__ == "__main__":
    main()
