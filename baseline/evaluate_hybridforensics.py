from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from common import (
    build_clip_classifier,
    build_mobilenet_v3_small,
    build_resnet18,
    build_transforms,
    evaluate,
    get_device,
    save_json,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class HybridForensicsDataset(Dataset):
    def __init__(self, root: Path, model_type: str):
        self.root = root
        self.transform = build_transforms(model_type=model_type, train=False)
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
            with Image.open(path) as image:
                image.verify()
            return True
        except (OSError, UnidentifiedImageError):
            self.skipped_files.append(str(path))
            return False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the HybridForensics benchmark."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="HybridForensics raw root. Defaults to data/hybridforensics/raw.",
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
    data_root = args.data_root or Path("data/hybridforensics/raw")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
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

    dataset = HybridForensicsDataset(root=data_root, model_type=transform_model_type)
    if dataset.skipped_files:
        print(f"Skipped unreadable files: {len(dataset.skipped_files)}")
        for path in dataset.skipped_files[:10]:
            print(f"  - {path}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    device = get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    result = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        positive_index=dataset.class_to_idx["fake"],
        progress_desc="eval hybridforensics",
    )

    payload = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(data_root),
        "benchmark": "hybridforensics",
        "model_type": model_type,
        "accuracy": result.accuracy,
        "auc": result.auc,
        "f1": result.f1,
        "loss": result.loss,
        "num_samples": len(dataset),
        "skipped_files": dataset.skipped_files,
    }
    print(payload)
    if args.save_json is not None:
        save_json(args.save_json, payload)


if __name__ == "__main__":
    main()
