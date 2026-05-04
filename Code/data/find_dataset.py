import os
import random
import csv
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".jpeg")
REAL_DIRS = ["nature", "real", "original", "0_real"]

train_pil_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
)

test_pil_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
)

stream_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ]
)

pil_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
)

to_tensor = transforms.ToTensor()
tensor_normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)


class FINDDataset(Dataset):
    def __init__(self, samples, pil_transform=None):
        self.samples = samples
        self.pil_transform = pil_transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            image = Image.open(item["data"]).convert("RGB")
            if self.pil_transform:
                image = self.pil_transform(image)
            tensor_img = self.to_tensor(image)
            return tensor_img, item["label"], item["type"]
        except Exception as exc:
            print(f"Error loading {item['data']}: {exc}")
            return self.__getitem__(random.randint(0, len(self) - 1))


class AIEvalDataset(Dataset):
    def __init__(self, file_paths: Iterable[str]):
        self.file_paths = list(file_paths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.file_paths[idx]).convert("RGB")
            img = pil_transforms(img)
            return tensor_normalize(to_tensor(img)), 1
        except Exception:
            return torch.zeros((3, 224, 224)), 1


def _build_balanced_samples(ai_paths, real_paths, max_ai_samples):
    random.shuffle(ai_paths)
    random.shuffle(real_paths)
    ai_paths = ai_paths[:max_ai_samples]
    sample_count = min(len(ai_paths), len(real_paths))
    ai_paths = ai_paths[:sample_count]
    real_paths = real_paths[:sample_count]

    samples = []
    for path in ai_paths:
        samples.append({"type": "ai_fake", "data": path, "label": 1})
    for path in real_paths:
        samples.append({"type": "real", "data": path, "label": 0})
    random.shuffle(samples)
    return samples


def _extract_generator_name(root_path, fake_path):
    relative_root = os.path.relpath(root_path, fake_path)
    parts = relative_root.replace("\\", "/").split("/")
    return parts[0] if parts and parts[0] != "." else "unknown"


def _select_held_out_generators(generator_names, fraction=0.25, seed=42):
    generator_names = sorted(set(generator_names))
    if not generator_names:
        return []

    rng = random.Random(seed)
    shuffled = generator_names[:]
    rng.shuffle(shuffled)
    holdout_count = max(1, int(round(len(shuffled) * fraction)))
    holdout_count = min(holdout_count, len(shuffled))
    return sorted(shuffled[:holdout_count])


def _flatten_grouped_paths(grouped_paths, generator_names):
    flattened = []
    for generator_name in generator_names:
        flattened.extend(grouped_paths.get(generator_name, []))
    return flattened


def _manifest_dir_for_data_path(fake_path):
    data_path = Path(fake_path)
    if data_path.name == "tiny-genimage":
        return data_path.parent / "manifests"
    return data_path / "manifests"


def _label_key(label):
    normalized = str(label).strip().lower()
    if normalized in {"0", "real"}:
        return "real"
    if normalized in {"1", "fake", "ai", "ai_fake"}:
        return "ai_fake"
    raise ValueError(f"Unsupported label in manifest: {label}")


def _load_manifest_rows(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return []

    project_root = manifest_path.resolve().parents[2]
    rows = []
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
            generator = (row.get("generator") or "unknown").strip() or "unknown"
            rows.append(
                {
                    "path": str(image_path),
                    "label_key": _label_key(label_text),
                    "generator": generator,
                }
            )
    return rows


def _load_manifest_split_files(fake_path, grouped=False):
    manifest_dir = _manifest_dir_for_data_path(fake_path)
    train_rows = _load_manifest_rows(manifest_dir / "train.csv")
    val_rows = _load_manifest_rows(manifest_dir / "val.csv")
    if not train_rows or not val_rows:
        return None

    if grouped:
        split_files = {
            "train": {"ai_fake": {}, "real": {}},
            "val": {"ai_fake": {}, "real": {}},
        }
        for split_name, rows in (("train", train_rows), ("val", val_rows)):
            for row in rows:
                split_files[split_name][row["label_key"]].setdefault(row["generator"], [])
                split_files[split_name][row["label_key"]][row["generator"]].append(row["path"])
        return split_files

    split_files = {
        "train": {"ai_fake": [], "real": []},
        "val": {"ai_fake": [], "real": []},
    }
    for split_name, rows in (("train", train_rows), ("val", val_rows)):
        for row in rows:
            split_files[split_name][row["label_key"]].append(row["path"])
    return split_files


def prepare_find_dataloaders(fake_path="data", max_ai_samples=10000, batch_size=96, num_workers=0):
    pin_memory = torch.cuda.is_available()
    split_files = _load_manifest_split_files(fake_path, grouped=False)
    if split_files is None:
        split_files = {
            "train": {"ai_fake": [], "real": []},
            "val": {"ai_fake": [], "real": []},
        }

        for root, dirs, files in os.walk(fake_path):
            dirs[:] = [name for name in dirs if not name.startswith(".")]
            path_parts = root.lower().replace("\\", "/").split("/")

            current_split = None
            if "train" in path_parts:
                current_split = "train"
            elif "val" in path_parts:
                current_split = "val"

            if current_split is None:
                continue

            is_real_folder = any(name in path_parts for name in REAL_DIRS)
            target_key = "real" if is_real_folder else "ai_fake"

            for filename in files:
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, filename)
                    split_files[current_split][target_key].append(full_path)

    train_data = _build_balanced_samples(
        split_files["train"]["ai_fake"],
        split_files["train"]["real"],
        max_ai_samples=max_ai_samples,
    )
    val_data = _build_balanced_samples(
        split_files["val"]["ai_fake"],
        split_files["val"]["real"],
        max_ai_samples=max_ai_samples,
    )

    train_dataset = FINDDataset(train_data, pil_transform=train_pil_transforms)
    val_dataset = FINDDataset(val_data, pil_transform=test_pil_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def prepare_find_dataloaders_with_ood(
    fake_path="data",
    max_ai_samples=10000,
    batch_size=96,
    num_workers=0,
    ood_generator_fraction=0.25,
    ood_seed=42,
):
    pin_memory = torch.cuda.is_available()
    split_files = _load_manifest_split_files(fake_path, grouped=True)
    if split_files is None:
        split_files = {
            "train": {"ai_fake": {}, "real": {}},
            "val": {"ai_fake": {}, "real": {}},
        }

        for root, dirs, files in os.walk(fake_path):
            dirs[:] = [name for name in dirs if not name.startswith(".")]
            path_parts = root.lower().replace("\\", "/").split("/")

            current_split = None
            if "train" in path_parts:
                current_split = "train"
            elif "val" in path_parts:
                current_split = "val"

            if current_split is None:
                continue

            generator_name = _extract_generator_name(root, fake_path)
            is_real_folder = any(name in path_parts for name in REAL_DIRS)
            target_key = "real" if is_real_folder else "ai_fake"

            split_files[current_split][target_key].setdefault(generator_name, [])
            for filename in files:
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    split_files[current_split][target_key][generator_name].append(os.path.join(root, filename))

    all_generators = set()
    for split_name in ("train", "val"):
        all_generators.update(split_files[split_name]["ai_fake"].keys())
        all_generators.update(split_files[split_name]["real"].keys())

    held_out_generators = _select_held_out_generators(
        all_generators,
        fraction=ood_generator_fraction,
        seed=ood_seed,
    )
    in_domain_generators = sorted(all_generators - set(held_out_generators))

    train_ai_paths = _flatten_grouped_paths(split_files["train"]["ai_fake"], in_domain_generators)
    train_real_paths = _flatten_grouped_paths(split_files["train"]["real"], in_domain_generators)
    val_ai_paths = _flatten_grouped_paths(split_files["val"]["ai_fake"], in_domain_generators)
    val_real_paths = _flatten_grouped_paths(split_files["val"]["real"], in_domain_generators)

    ood_ai_paths = _flatten_grouped_paths(split_files["train"]["ai_fake"], held_out_generators)
    ood_ai_paths.extend(_flatten_grouped_paths(split_files["val"]["ai_fake"], held_out_generators))
    ood_real_paths = _flatten_grouped_paths(split_files["train"]["real"], held_out_generators)
    ood_real_paths.extend(_flatten_grouped_paths(split_files["val"]["real"], held_out_generators))

    train_data = _build_balanced_samples(train_ai_paths, train_real_paths, max_ai_samples=max_ai_samples)
    val_data = _build_balanced_samples(val_ai_paths, val_real_paths, max_ai_samples=max_ai_samples)
    ood_val_data = _build_balanced_samples(ood_ai_paths, ood_real_paths, max_ai_samples=max_ai_samples)

    if not train_data:
        raise RuntimeError("No training samples found after OOD generator split.")
    if not val_data:
        raise RuntimeError("No in-domain validation samples found after OOD generator split.")
    if not ood_val_data:
        raise RuntimeError(
            "No OOD validation samples found after OOD generator split. "
            "Try a different --ood-generator-fraction or --ood-seed."
        )

    train_dataset = FINDDataset(train_data, pil_transform=train_pil_transforms)
    val_dataset = FINDDataset(val_data, pil_transform=test_pil_transforms)
    ood_val_dataset = FINDDataset(ood_val_data, pil_transform=test_pil_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    ood_val_loader = DataLoader(
        ood_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, ood_val_loader, held_out_generators
