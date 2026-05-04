from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from common import (
    CLASS_TO_IDX,
    IMAGE_EXTENSIONS,
    build_model,
    build_transforms,
    get_device,
    save_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ImageFileDataset(Dataset):
    def __init__(self, paths: list[Path], label: int, model_type: str):
        self.paths = paths
        self.label = label
        self.transform = build_transforms(model_type=model_type, train=False)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), self.label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a baseline checkpoint on MNW.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mnw-dir", type=Path, default=PROJECT_ROOT / "data" / "MNW" / "AI_Images")
    parser.add_argument("--test-manifest", type=Path, default=PROJECT_ROOT / "data" / "manifests" / "test.csv")
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=None,
        help="Optional local real-image directory. If omitted, ImageNet validation streaming is used.",
    )
    parser.add_argument("--num-real-samples", type=int, default=15260)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-fpr", type=float, default=None)
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument("--hf-token", default=None)
    return parser.parse_args()


def iter_image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_test_manifest(manifest_path: Path) -> dict[str, list[Path]]:
    if not manifest_path.exists():
        return {}

    generator_paths: dict[str, list[Path]] = {}
    project_root = manifest_path.resolve().parents[2]
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            path_text = (row.get("path") or "").strip()
            label_text = (row.get("label") or "").strip().lower()
            if not path_text or label_text not in {"fake", "1", "ai", "ai_fake"}:
                continue
            image_path = Path(path_text)
            if not image_path.is_absolute():
                image_path = project_root / image_path
            generator = (row.get("generator") or "unknown").strip() or "unknown"
            generator_paths.setdefault(generator, []).append(image_path)
    return generator_paths


def get_hf_token(explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token
    for key in ("HF_TOKEN", "TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        value = os.getenv(key)
        if value:
            return value
    return None


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    fake_index: int,
    desc: str,
) -> tuple[list[int], list[float]]:
    model.eval()
    labels_out: list[int] = []
    probs_out: list[float] = []
    for images, labels in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, fake_index]
        labels_out.extend(labels.tolist())
        probs_out.extend(probs.detach().cpu().tolist())
    return labels_out, probs_out


@torch.no_grad()
def predict_imagenet_stream(
    model: nn.Module,
    device: torch.device,
    fake_index: int,
    model_type: str,
    token: str,
    num_samples: int,
    batch_size: int,
) -> tuple[list[int], list[float]]:
    transform = build_transforms(model_type=model_type, train=False)
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=token)
    labels_out: list[int] = []
    probs_out: list[float] = []
    batch: list[torch.Tensor] = []

    pbar = tqdm(total=num_samples, desc="ImageNet real", leave=False, dynamic_ncols=True)
    for item in dataset:
        if len(labels_out) >= num_samples:
            break
        image = item["image"]
        if image.mode != "RGB":
            continue
        batch.append(transform(image))
        if len(batch) == batch_size:
            images = torch.stack(batch).to(device)
            probs = torch.softmax(model(images), dim=1)[:, fake_index]
            probs_out.extend(probs.detach().cpu().tolist())
            labels_out.extend([CLASS_TO_IDX["real"]] * len(batch))
            pbar.update(len(batch))
            batch = []

    if batch:
        images = torch.stack(batch).to(device)
        probs = torch.softmax(model(images), dim=1)[:, fake_index]
        probs_out.extend(probs.detach().cpu().tolist())
        labels_out.extend([CLASS_TO_IDX["real"]] * len(batch))
        pbar.update(len(batch))
    pbar.close()
    return labels_out, probs_out


def threshold_for_target_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    fpr, _, thresholds = roc_curve(y_true, y_score, pos_label=CLASS_TO_IDX["fake"])
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return float(thresholds[0])
    return float(thresholds[valid[-1]])


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_type = checkpoint["model_type"]
    model_type_for_transform = "clip" if model_type == "clip" else "cnn"
    fake_index = checkpoint.get("class_to_idx", CLASS_TO_IDX)["fake"]

    model = build_model(
        model_type=model_type,
        pretrained=False,
        clip_model_name=checkpoint.get("clip_model_name", "openai/clip-vit-base-patch32"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = get_device(args.device)
    model = model.to(device)
    model.eval()

    y_true: list[int] = []
    y_score: list[float] = []

    if args.real_dir is not None:
        real_paths = iter_image_paths(args.real_dir)[: args.num_real_samples]
        real_loader = DataLoader(
            ImageFileDataset(real_paths, CLASS_TO_IDX["real"], model_type_for_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        labels, probs = predict_loader(model, real_loader, device, fake_index, "local real")
    else:
        token = get_hf_token(args.hf_token)
        if token is None:
            raise RuntimeError(
                "MNW evaluation needs real images. Provide --real-dir or set HF_TOKEN for ImageNet validation."
            )
        labels, probs = predict_imagenet_stream(
            model=model,
            device=device,
            fake_index=fake_index,
            model_type=model_type_for_transform,
            token=token,
            num_samples=args.num_real_samples,
            batch_size=args.batch_size,
        )
    y_true.extend(labels)
    y_score.extend(probs)

    generator_probs: dict[str, list[float]] = {}
    generator_path_map = load_test_manifest(args.test_manifest)
    if not generator_path_map:
        generator_path_map = {
            path.name: iter_image_paths(path)
            for path in sorted(args.mnw_dir.iterdir())
            if path.is_dir()
        }

    for generator_name, fake_paths in sorted(generator_path_map.items()):
        loader = DataLoader(
            ImageFileDataset(fake_paths, CLASS_TO_IDX["fake"], model_type_for_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        labels, probs = predict_loader(model, loader, device, fake_index, generator_name)
        y_true.extend(labels)
        y_score.extend(probs)
        generator_probs[generator_name] = probs

    y_true_array = np.asarray(y_true, dtype=np.int64)
    y_score_array = np.asarray(y_score, dtype=np.float32)
    threshold = args.threshold
    if args.target_fpr is not None:
        threshold = threshold_for_target_fpr(y_true_array, y_score_array, args.target_fpr)
    y_pred_array = (y_score_array > threshold).astype(np.int64)
    cm = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1])
    generator_results: dict[str, dict[str, float | int]] = {}
    for generator_name, probs in generator_probs.items():
        predictions = [int(prob > threshold) for prob in probs]
        generator_results[generator_name] = {
            "total": len(probs),
            "detected_fake": int(sum(predictions)),
            "recall": float(sum(predictions) / max(len(predictions), 1)),
        }

    payload = {
        "checkpoint": str(args.checkpoint),
        "model": model_type,
        "test_dataset": "MNW",
        "mnw_dir": str(args.mnw_dir),
        "real_source": str(args.real_dir) if args.real_dir is not None else "ILSVRC/imagenet-1k validation",
        "label_convention": CLASS_TO_IDX,
        "threshold": threshold,
        "target_fpr": args.target_fpr,
        "metrics": {
            "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
            "auc": float(roc_auc_score(y_true_array, y_score_array)),
            "f1": float(f1_score(y_true_array, y_pred_array, pos_label=1, zero_division=0)),
            "confusion_matrix_real_fake": cm.tolist(),
        },
        "num_samples": {
            "real": int((y_true_array == 0).sum()),
            "fake": int((y_true_array == 1).sum()),
            "total": int(len(y_true_array)),
        },
        "generator_results": generator_results,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.save_json is not None:
        save_json(args.save_json, payload)


if __name__ == "__main__":
    main()
