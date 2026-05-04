import json
import os
import csv
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import HF_TOKEN_ENV_KEYS, PROJECT_ROOT
from data.find_dataset import AIEvalDataset, IMAGE_EXTENSIONS, stream_transforms


def get_hf_token(explicit_token: Optional[str] = None):
    if explicit_token:
        return explicit_token
    for key in HF_TOKEN_ENV_KEYS:
        value = os.getenv(key)
        if value:
            return value
    return None


def require_hf_token(explicit_token: Optional[str] = None):
    token = get_hf_token(explicit_token)
    if token is not None:
        return token
    raise RuntimeError(
        "Hugging Face token not found. Set one of HF_TOKEN / TOKEN / "
        "HUGGINGFACE_HUB_TOKEN / HF_HUB_TOKEN, or run `hf auth login` first."
    )


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(preferred="auto"):
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred != "auto":
        print(f"Requested device '{preferred}' is unavailable. Falling back to auto selection.")
    return get_default_device()


def describe_device(device):
    device = torch.device(device)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        return f"cuda:{index} ({name})"
    return str(device)


def print_device_banner(device):
    device_text = describe_device(device)
    print(f"Using device: {device_text}")
    if torch.device(device).type == "cuda":
        torch.backends.cudnn.benchmark = True


def resolve_output_path(path, output_dir="."):
    if os.path.isabs(path):
        return path
    return os.path.join(output_dir, path)


def save_test_results(result, save_path="test_results.npz", output_dir="."):
    resolved_path = resolve_output_path(save_path, output_dir=output_dir)
    os.makedirs(os.path.dirname(resolved_path) or output_dir, exist_ok=True)
    np.savez(
        resolved_path,
        y_true=np.asarray(result["y_true"], dtype=np.int64),
        y_probs=np.asarray(result["y_probs"], dtype=np.float32),
    )
    return resolved_path


def save_evaluation_reports(reports, save_path="evaluation_reports.json", output_dir="."):
    resolved_path = resolve_output_path(save_path, output_dir=output_dir)
    os.makedirs(os.path.dirname(resolved_path) or output_dir, exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2, ensure_ascii=False)
    return resolved_path


def get_writable_checkpoint_path(filename, output_dir="."):
    base_path = resolve_output_path(filename, output_dir=output_dir)
    if not os.path.exists(base_path):
        return base_path

    base_dir = os.path.dirname(base_path) or output_dir
    stem, ext = os.path.splitext(os.path.basename(base_path))
    for run_idx in range(1, 1000):
        candidate = os.path.join(base_dir, f"{stem}_run{run_idx}{ext}")
        if not os.path.exists(candidate):
            return candidate

    raise RuntimeError(f"No writable checkpoint path available for {filename}")


def save_checkpoint(model, target_path, output_dir=".", record_path="latest_checkpoint.txt"):
    os.makedirs(output_dir, exist_ok=True)
    save_path = get_writable_checkpoint_path(target_path, output_dir=output_dir)
    torch.save(model.state_dict(), save_path)
    with open(resolve_output_path(record_path, output_dir=output_dir), "w", encoding="utf-8") as handle:
        handle.write(save_path)
    return save_path


def resolve_checkpoint_path(default_path, output_dir=".", record_path=None):
    if record_path:
        record_file = resolve_output_path(record_path, output_dir=output_dir)
        if os.path.exists(record_file):
            recorded_path = open(record_file, "r", encoding="utf-8").read().strip()
            if recorded_path and os.path.exists(recorded_path):
                return recorded_path

    candidate = resolve_output_path(default_path, output_dir=output_dir)
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(f"Checkpoint not found: {candidate}")


def frequency_domain_masking(images, mask_ratio=0.15):
    fft_images = torch.fft.fft2(images, dim=(-2, -1))
    fft_shift = torch.fft.fftshift(fft_images, dim=(-2, -1))
    batch_size, _, height, width = images.shape
    mask = torch.rand(batch_size, 1, height, width, device=images.device) > mask_ratio
    fft_shift_masked = fft_shift * mask
    fft_masked = torch.fft.ifftshift(fft_shift_masked, dim=(-2, -1))
    images_masked = torch.fft.ifft2(fft_masked, dim=(-2, -1)).real
    return torch.clamp(images_masked, 0.0, 1.0)


def load_mnw_manifest(manifest_path=None):
    manifest_path = manifest_path or os.path.join(PROJECT_ROOT, "data", "manifests", "test.csv")
    if not os.path.exists(manifest_path):
        return {}

    generator_paths = {}
    with open(manifest_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            path_text = (row.get("path") or "").strip()
            label_text = (row.get("label") or "").strip().lower()
            if not path_text or label_text not in {"fake", "1", "ai", "ai_fake"}:
                continue
            if not os.path.isabs(path_text):
                path_text = os.path.join(PROJECT_ROOT, path_text)
            generator = (row.get("generator") or "unknown").strip() or "unknown"
            generator_paths.setdefault(generator, []).append(path_text)
    return generator_paths


def test_mnw_evaluation(
    model,
    device,
    token=None,
    val_dir=None,
    num_real_test_samples=15260,
    batch_size=64,
    threshold=0.5,
):
    result = collect_mnw_evaluation_outputs(
        model,
        device=device,
        token=token,
        val_dir=val_dir,
        num_real_test_samples=num_real_test_samples,
        batch_size=batch_size,
    )
    return summarize_mnw_evaluation(result, threshold=threshold)


def collect_mnw_evaluation_outputs(
    model,
    device,
    token=None,
    val_dir=None,
    num_real_test_samples=15260,
    batch_size=64,
):
    token = require_hf_token(token)
    if val_dir is None:
        val_dir = os.path.join(PROJECT_ROOT, "data", "MNW", "AI_Images")

    all_y_true = []
    all_y_probs = []
    real_probs = []
    generator_probs = {}

    print("\nStarting evaluation inference")

    print(f"\n--- Phase 1: Evaluating Real Images (Target: {num_real_test_samples}) ---")
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=token)
    batch_tensors = []
    total_real_processed = 0

    pbar_real = tqdm(total=num_real_test_samples, desc="ImageNet Reals")
    with torch.no_grad():
        for item in dataset:
            if total_real_processed >= num_real_test_samples:
                break

            img = item["image"]
            if img.mode != "RGB":
                continue

            batch_tensors.append(stream_transforms(img))
            total_real_processed += 1
            pbar_real.update(1)

            if len(batch_tensors) == batch_size or total_real_processed == num_real_test_samples:
                batch_data = torch.stack(batch_tensors).to(device)
                outputs = model(batch_data)
                probs = torch.softmax(outputs, dim=1)

                prob_class_1 = probs[:, 1]
                batch_probs = prob_class_1.cpu().tolist()
                real_probs.extend(batch_probs)
                all_y_probs.extend(batch_probs)
                all_y_true.extend([0] * len(batch_probs))
                batch_tensors = []
    pbar_real.close()

    print("\n--- Phase 2: Evaluating AI-Generated Images ---")
    generator_path_map = load_mnw_manifest()
    if not generator_path_map:
        generators = [name for name in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, name))]
        generators.sort()
        generator_path_map = {
            gen_name: [
                os.path.join(root, filename)
                for root, _, filenames in os.walk(os.path.join(val_dir, gen_name))
                for filename in filenames
                if filename.lower().endswith(IMAGE_EXTENSIONS)
            ]
            for gen_name in generators
        }

    for gen_name, img_paths in sorted(generator_path_map.items()):
        total_imgs = len(img_paths)
        if total_imgs == 0:
            continue

        dataset = AIEvalDataset(img_paths)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        gen_probs = []
        pbar_gen = tqdm(loader, desc=f"{gen_name[:15]:<15}")
        with torch.no_grad():
            for imgs, lbls in pbar_gen:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)

                prob_class_1 = probs[:, 1]
                batch_probs = prob_class_1.cpu().tolist()
                gen_probs.extend(batch_probs)
                all_y_probs.extend(batch_probs)
                all_y_true.extend(lbls.cpu().tolist())

        generator_probs[gen_name] = {
            "probs": gen_probs,
            "total": total_imgs,
        }

    return {
        "real_probs": real_probs,
        "generator_probs": generator_probs,
        "y_true": all_y_true,
        "y_probs": all_y_probs,
    }


def summarize_mnw_evaluation(result, threshold=0.5, report_title="Joint Evaluation Final Report"):
    tn = sum(prob <= threshold for prob in result["real_probs"])
    fp = sum(prob > threshold for prob in result["real_probs"])
    tp = 0
    fn = 0
    generator_results = {}

    for gen_name, gen_data in result["generator_probs"].items():
        gen_tp = sum(prob > threshold for prob in gen_data["probs"])
        gen_fn = gen_data["total"] - gen_tp
        generator_results[gen_name] = {"TP": gen_tp, "total": gen_data["total"]}
        tp += gen_tp
        fn += gen_fn

    epsilon = 1e-7
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    fpr = fp / (fp + tn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    print(f"\nStarting evaluation report | threshold: {threshold:.6f}")
    print("\n" + "=" * 60)
    print(report_title)
    print("=" * 60)
    print(f"Accuracy:                  {accuracy * 100:.2f}%")
    print(f"Recall (AI Fake Detection):{recall * 100:.2f}%")
    print(f"False Positive Rate:       {fpr * 100:.2f}%")
    print(f"Precision:                 {precision * 100:.2f}%")
    print(f"F1-Score:                  {f1_score:.4f}")
    print("-" * 60)
    print(f"TP: {tp} | TN: {tn} | FN: {fn} | FP: {fp}")
    print("-" * 60)
    print("Recall by Generator:")
    for gen, metrics in generator_results.items():
        gen_acc = 100.0 * metrics["TP"] / metrics["total"]
        print(f"  - {gen:<25}: {gen_acc:>6.2f}% ({metrics['TP']}/{metrics['total']})")
    print("=" * 60)

    return {
        "report_title": report_title,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1_score": f1_score,
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fn": fn,
            "fp": fp,
        },
        "generator_results": generator_results,
        "y_true": result["y_true"],
        "y_probs": result["y_probs"],
        "threshold": threshold,
    }


def plot_roc_with_target_fpr(y_true, y_probs, target_fpr=0.1, save_path="roc.png"):
    import matplotlib.pyplot as plt

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    reference_fprs = [0.01, 0.05, 0.10]
    selected_points = []

    for ref_fpr in reference_fprs:
        valid_indices = np.where(fpr <= ref_fpr)[0]
        if len(valid_indices) == 0:
            continue

        best_index = valid_indices[-1]
        point = {
            "target_fpr": ref_fpr,
            "threshold": thresholds[best_index],
            "actual_fpr": fpr[best_index],
            "actual_tpr": tpr[best_index],
        }
        selected_points.append(point)

        print("\n" + "=" * 50)
        print(f"Target FPR <= {ref_fpr}")
        print(f"Recommended threshold: {point['threshold']:.4f}")
        print(f"Actual FPR: {point['actual_fpr']:.4%} | Expected TPR: {point['actual_tpr']:.4%}")
        print("=" * 50)

    if not selected_points:
        print("No threshold satisfies the reference FPR targets.")

    plt.figure(figsize=(9, 7))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess")

    marker_colors = ["purple", "red", "green"]
    for color, point in zip(marker_colors, selected_points):
        plt.plot(
            point["actual_fpr"],
            point["actual_tpr"],
            marker="o",
            markersize=8,
            color=color,
            label=(
                f"FPR <= {point['target_fpr']:.2f}\n"
                f"Threshold = {point['threshold']:.4f}"
            ),
        )
        plt.vlines(
            x=point["actual_fpr"],
            ymin=0,
            ymax=point["actual_tpr"],
            colors=color,
            linestyles="dotted",
            alpha=0.7,
        )
        plt.hlines(
            y=point["actual_tpr"],
            xmin=0,
            xmax=point["actual_fpr"],
            colors=color,
            linestyles="dotted",
            alpha=0.7,
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve with FPR 0.01, 0.05, and 0.10 Operating Points")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nROC saved to: {save_path}")
    plt.show()

    for point in selected_points:
        if abs(point["target_fpr"] - target_fpr) < 1e-9:
            return point["threshold"]
    return selected_points[-1]["threshold"] if selected_points else None
