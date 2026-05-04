from __future__ import annotations

import json
import random
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModel


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_DIR_NAMES = {"real", "nature", "original", "0_real", "imagenet"}
FAKE_DIR_NAMES = {"fake", "ai", "ai_fake", "generated", "1_fake"}
CLASS_TO_IDX = {"real": 0, "fake": 1}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    auc: float
    f1: float


class ImagePathDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform
        self.class_to_idx = CLASS_TO_IDX.copy()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


class FrozenCLIPClassifier(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", num_classes: int = 2):
        super().__init__()
        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=pixel_values)
        return self.classifier(outputs.pooler_output)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred != "auto":
        print(f"Requested device '{preferred}' is unavailable. Falling back to auto selection.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(model_type: str, train: bool) -> transforms.Compose:
    if model_type == "clip":
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    if train:
        pil_ops: list[Any] = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        pil_ops = [transforms.Resize(256), transforms.CenterCrop(224)]

    return transforms.Compose(
        pil_ops + [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )


def _iter_image_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _label_from_path(path: Path, split_root: Path) -> int | None:
    parts = {part.lower() for part in path.relative_to(split_root).parts[:-1]}
    if parts & REAL_DIR_NAMES:
        return CLASS_TO_IDX["real"]
    if parts & FAKE_DIR_NAMES:
        return CLASS_TO_IDX["fake"]

    # Tiny-GenImage usually stores real images under a nature/original branch.
    # If no explicit real marker is present, generator folders are treated as fake.
    if len(path.relative_to(split_root).parts) > 1:
        return CLASS_TO_IDX["fake"]
    return None


def collect_split_samples(data_root: Path, split: str) -> list[tuple[Path, int]]:
    split_root = data_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    samples: list[tuple[Path, int]] = []
    skipped: list[Path] = []
    for path in _iter_image_files(split_root):
        label = _label_from_path(path, split_root)
        if label is None:
            skipped.append(path)
            continue
        samples.append((path, label))

    if not samples:
        raise RuntimeError(f"No images found under {split_root}")

    labels = {label for _, label in samples}
    if labels != {CLASS_TO_IDX["real"], CLASS_TO_IDX["fake"]}:
        raise RuntimeError(
            f"{split_root} must contain both real and fake images. Found labels: {sorted(labels)}"
        )

    return sorted(samples, key=lambda item: str(item[0]))


def _label_to_index(label: str) -> int:
    normalized = label.strip().lower()
    if normalized in {"0", "real"}:
        return CLASS_TO_IDX["real"]
    if normalized in {"1", "fake", "ai", "ai_fake"}:
        return CLASS_TO_IDX["fake"]
    raise ValueError(f"Unsupported label in manifest: {label}")


def collect_manifest_samples(manifest_path: Path) -> list[tuple[Path, int]]:
    if not manifest_path.exists():
        return []

    project_root = manifest_path.resolve().parents[2]
    samples: list[tuple[Path, int]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            path_text = (row.get("path") or "").strip()
            label_text = (row.get("label") or "").strip()
            if not path_text or not label_text:
                continue
            image_path = Path(path_text)
            if not image_path.is_absolute():
                image_path = project_root / image_path
            samples.append((image_path, _label_to_index(label_text)))
    return samples


def _sample_indices_per_class(samples: list[tuple[Path, int]], per_class: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    grouped: dict[int, list[int]] = {}
    for index, (_, label) in enumerate(samples):
        grouped.setdefault(label, []).append(index)

    selected: list[int] = []
    for label, indices in grouped.items():
        if len(indices) < per_class:
            raise ValueError(f"label {label} only has {len(indices)} samples, need {per_class}")
        rng.shuffle(indices)
        selected.extend(indices[:per_class])
    return sorted(selected)


def build_loader(
    data_root: Path,
    split: str,
    model_type: str,
    batch_size: int,
    num_workers: int,
    manifest_path: Path | None = None,
    subset_per_class: int | None = None,
    seed: int = 4210,
) -> tuple[DataLoader, dict[str, int]]:
    samples = collect_manifest_samples(manifest_path) if manifest_path is not None else []
    if not samples:
        samples = collect_split_samples(data_root, split)
    dataset: Dataset = ImagePathDataset(
        samples=samples,
        transform=build_transforms(model_type=model_type, train=(split == "train")),
    )
    if split == "train" and subset_per_class is not None:
        dataset = Subset(dataset, _sample_indices_per_class(samples, subset_per_class, seed))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, CLASS_TO_IDX.copy()


def build_model(model_type: str, pretrained: bool = True, clip_model_name: str = "openai/clip-vit-base-patch32") -> nn.Module:
    if model_type == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    if model_type == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        last_linear = model.classifier[-1]
        if not isinstance(last_linear, nn.Linear):
            raise TypeError("Expected MobileNetV3 classifier head to end with nn.Linear.")
        model.classifier[-1] = nn.Linear(last_linear.in_features, 2)
        return model

    if model_type == "clip":
        return FrozenCLIPClassifier(model_name=clip_model_name, num_classes=2)

    raise ValueError(f"Unsupported model type: {model_type}")


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, float]:
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    y_true = labels.detach().cpu().numpy()
    y_pred = predictions.detach().cpu().numpy()
    y_score = probs[:, CLASS_TO_IDX["fake"]].detach().cpu().numpy()

    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, pos_label=CLASS_TO_IDX["fake"], zero_division=0))
    auc = float(roc_auc_score(y_true, y_score))
    return accuracy, auc, f1


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    progress_desc: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in tqdm(loader, desc=progress_desc, leave=False, dynamic_ncols=True):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    progress_desc: str,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in tqdm(loader, desc=progress_desc, leave=False, dynamic_ncols=True):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    accuracy, auc, f1 = compute_metrics(logits_tensor, labels_tensor)
    return EvalResult(loss=total_loss / max(total_samples, 1), accuracy=accuracy, auc=auc, f1=f1)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_checkpoint(path: Path, model: nn.Module, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **payload}, path)
