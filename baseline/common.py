from __future__ import annotations

import json
import random
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from torchvision import datasets, models, transforms
from transformers import CLIPVisionModel


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


class FrozenCLIPClassifier(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", num_classes: int = 2):
        super().__init__()
        self.model_name = model_name
        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        return self.classifier(pooled)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def to_record_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(get_project_root()))
    except ValueError:
        return str(path)


def get_current_command() -> str:
    return " ".join(shlex.quote(arg) for arg in sys.argv)


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=get_project_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "需手动填写"


def load_experiment_record_template() -> dict[str, Any]:
    template_path = get_project_root() / "实验记录要求" / "实验记录模板.json"
    with template_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def result_to_metrics(result: EvalResult | None) -> dict[str, Any]:
    if result is None:
        return {
            "accuracy": None,
            "auc": None,
            "f1": None,
            "loss": None,
        }
    return {
        "accuracy": result.accuracy,
        "auc": result.auc,
        "f1": result.f1,
        "loss": result.loss,
    }


def build_record_base(
    *,
    record_type: str,
    stage: str,
    tags: list[str],
    seed: int,
    device: str | torch.device,
    script_path: str,
) -> dict[str, Any]:
    record = load_experiment_record_template()
    record["record_type"] = record_type
    record["experiment_id"] = "需手动填写"
    record["run_id"] = "需手动填写"
    record["date"] = date.today().isoformat()
    record["owner"] = "需手动填写"
    record["stage"] = stage
    record["tags"] = tags
    record["include_in_report"] = False
    record["include_in_ppt"] = False
    record["git_commit"] = get_git_commit()
    record["script_path"] = script_path
    record["command"] = get_current_command()
    record["seed"] = seed
    record["device"] = str(device)
    record["purpose"] = "需手动填写"
    record["hypothesis"] = "需手动填写"
    record["compare_to"] = ["需手动填写"]
    record["summary"]["main_finding"] = "需手动填写"
    record["summary"]["failure_mode"] = "需手动填写"
    record["summary"]["next_action"] = "需手动填写"
    record["summary"]["report_sentence"] = "需手动填写"
    return record


def build_evaluation_entry(
    *,
    checkpoint_path: Path,
    eval_type: str,
    data_root: Path,
    split: str,
    perturbation_type: str,
    result: EvalResult,
    num_samples: int,
    skipped_files: list[str] | None = None,
    jpeg_quality: int | None = None,
    resize_setting: str | None = None,
    note: str = "",
    enabled: bool = True,
) -> dict[str, Any]:
    return {
        "eval_id": "需手动填写",
        "enabled": enabled,
        "checkpoint_path": to_record_path(checkpoint_path),
        "eval_type": eval_type,
        "data_root": to_record_path(data_root),
        "split": split,
        "perturbation": {
            "type": perturbation_type,
            "jpeg_quality": jpeg_quality,
            "resize_setting": resize_setting,
        },
        "metrics": result_to_metrics(result),
        "num_samples": num_samples,
        "skipped_files": skipped_files or [],
        "note": note or "需手动填写",
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(model_type: str, train: bool) -> transforms.Compose:
    if model_type == "clip":
        mean = CLIP_MEAN
        std = CLIP_STD
    else:
        mean = IMAGENET_MEAN
        std = IMAGENET_STD

    ops: list[Any] = [transforms.Resize(256), transforms.CenterCrop(224)]
    if train:
        ops.insert(1, transforms.RandomHorizontalFlip(p=0.5))
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)


def _sample_indices_per_class(
    targets: list[int], subset_per_class: int, seed: int
) -> list[int]:
    rng = random.Random(seed)
    class_to_indices: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        class_to_indices.setdefault(target, []).append(index)

    sampled_indices: list[int] = []
    for class_id, indices in sorted(class_to_indices.items()):
        if len(indices) < subset_per_class:
            raise ValueError(
                f"class {class_id} only has {len(indices)} samples, "
                f"but {subset_per_class} are required"
            )
        rng.shuffle(indices)
        sampled_indices.extend(indices[:subset_per_class])
    sampled_indices.sort()
    return sampled_indices


def create_dataset(
    data_root: Path,
    split: str,
    model_type: str,
    train_subset_per_class: int | None = None,
    seed: int = 4210,
):
    dataset = datasets.ImageFolder(
        root=str(data_root / split),
        transform=build_transforms(model_type=model_type, train=(split == "train")),
    )
    if split == "train" and train_subset_per_class is not None:
        indices = _sample_indices_per_class(
            targets=dataset.targets,
            subset_per_class=train_subset_per_class,
            seed=seed,
        )
        dataset = Subset(dataset, indices)
        dataset.class_to_idx = {"fake": 0, "real": 1}  # type: ignore[attr-defined]
    return dataset


def build_loader(
    data_root: Path,
    split: str,
    model_type: str,
    batch_size: int,
    num_workers: int,
    train_subset_per_class: int | None = None,
    seed: int = 4210,
) -> tuple[DataLoader, dict[str, int]]:
    dataset = create_dataset(
        data_root=data_root,
        split=split,
        model_type=model_type,
        train_subset_per_class=train_subset_per_class,
        seed=seed,
    )
    class_to_idx = getattr(dataset, "class_to_idx", {"fake": 0, "real": 1})
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, class_to_idx


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_mobilenet_v3_small(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    last_linear = model.classifier[-1]
    if not isinstance(last_linear, nn.Linear):
        raise TypeError("Expected MobileNetV3 classifier head to end with nn.Linear.")
    model.classifier[-1] = nn.Linear(last_linear.in_features, num_classes)
    return model


def build_clip_classifier(model_name: str, num_classes: int = 2) -> nn.Module:
    return FrozenCLIPClassifier(model_name=model_name, num_classes=num_classes)


def compute_metrics(
    logits: torch.Tensor, labels: torch.Tensor, positive_index: int
) -> tuple[float, float, float]:
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    y_true = labels.detach().cpu().numpy()
    y_pred = predictions.detach().cpu().numpy()
    y_score = probs[:, positive_index].detach().cpu().numpy()
    y_binary = (y_true == positive_index).astype(np.int32)

    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_binary, y_pred == positive_index, zero_division=0))
    auc = float(roc_auc_score(y_binary, y_score))
    return accuracy, auc, f1


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    progress_desc: str | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc=progress_desc or "train",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels in progress:
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
        avg_loss = total_loss / max(total_samples, 1)
        progress.set_postfix(loss=f"{avg_loss:.4f}")

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    positive_index: int,
    progress_desc: str | None = None,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    progress = tqdm(
        loader,
        desc=progress_desc or "eval",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        avg_loss = total_loss / max(total_samples, 1)
        progress.set_postfix(loss=f"{avg_loss:.4f}")

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    accuracy, auc, f1 = compute_metrics(
        logits=logits_tensor,
        labels=labels_tensor,
        positive_index=positive_index,
    )
    return EvalResult(
        loss=total_loss / max(total_samples, 1),
        accuracy=accuracy,
        auc=auc,
        f1=f1,
    )


def save_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    payload: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **payload}, output_path)
