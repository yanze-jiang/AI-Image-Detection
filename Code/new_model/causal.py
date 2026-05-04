import csv
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve
from tqdm import tqdm
from transformers import CLIPVisionModel

from .config import (
    CAUSAL_CHECKPOINT_RECORD_PATH,
    CAUSAL_CLIP_MODEL,
    CAUSAL_EPOCHS,
    CAUSAL_LOG_PATH,
    CAUSAL_LR,
    CAUSAL_MODEL_SAVE_PATH,
    CAUSAL_NOISE_STD,
    DATA_DIR,
    DEFAULT_MAX_AI_SAMPLES,
    DEFAULT_TRAIN_BATCH_SIZE,
    OUTPUT_DIR,
)
from data.find_dataset import prepare_find_dataloaders_with_ood, tensor_normalize
from .utils import (
    frequency_domain_masking,
    get_default_device,
    get_hf_token,
    print_device_banner,
    resolve_checkpoint_path,
    resolve_output_path,
    save_checkpoint,
)


@dataclass
class CausalConfig:
    device: torch.device = field(default_factory=get_default_device)
    clip_model: str = CAUSAL_CLIP_MODEL
    epochs: int = CAUSAL_EPOCHS
    lr: float = CAUSAL_LR
    noise_std: float = CAUSAL_NOISE_STD
    data_dir: str = DATA_DIR
    max_ai_samples: int = DEFAULT_MAX_AI_SAMPLES
    batch_size: int = DEFAULT_TRAIN_BATCH_SIZE
    model_save_path: str = CAUSAL_MODEL_SAVE_PATH
    csv_log_path: str = CAUSAL_LOG_PATH
    checkpoint_record_path: str = CAUSAL_CHECKPOINT_RECORD_PATH
    output_dir: str = OUTPUT_DIR
    token: str | None = None
    causal_mode: str = "full"
    ood_generator_fraction: float = 0.25
    ood_seed: int = 42


class CausalCLIPHead(nn.Module):
    def __init__(
        self,
        feature_dim=768,
        tau=1.0,
        p_drop=0.3,
        scale_factor=15.0,
    ):
        super().__init__()
        self.tau = tau
        self.p_drop = p_drop
        self.scale_factor = scale_factor
        self.mask_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
        )
        self.classifier_h = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
        self.adversary_d = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def generate_mask(self, embedding):
        logits = self.mask_mlp(embedding)
        if self.training:
            gumbel_noise = -torch.empty_like(logits).exponential_().log()
            return torch.sigmoid((logits + gumbel_noise) / self.tau)
        return torch.sigmoid(logits / self.tau)

    def forward(self, embedding):
        mask = self.generate_mask(embedding)
        causal_features = mask * embedding
        non_causal_features = (1.0 - mask) * embedding
        logits_h = self.classifier_h(causal_features) * self.scale_factor

        if not self.training:
            return logits_h

        logits_d = self.adversary_d(non_causal_features)
        logits_d_detached = self.adversary_d(non_causal_features.detach())
        binary_mask = (torch.rand_like(causal_features) > self.p_drop).float()
        counterfactual_features = causal_features * binary_mask
        logits_h_cf = self.classifier_h(counterfactual_features) * self.scale_factor
        return logits_h, logits_d, mask, logits_h_cf, logits_d_detached


class CausalFINDClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.causal_head = CausalCLIPHead(feature_dim=backbone.config.hidden_size, tau=2.0)

    def forward(self, x):
        outputs = self.backbone(x)
        hidden_states = outputs.hidden_states
        layer_8_features = hidden_states[8]
        cls_token = layer_8_features[:, 0, :]
        cls_token = F.normalize(cls_token, p=2, dim=1)
        return self.causal_head(cls_token)


def build_causal_model(config: CausalConfig):
    print_device_banner(config.device)
    token = get_hf_token(config.token)
    visual_backbone = CLIPVisionModel.from_pretrained(config.clip_model, output_hidden_states=True, token=token)
    for param in visual_backbone.parameters():
        param.requires_grad = False
    target_layers = [5, 6, 7, 8]
    for index in target_layers:
        for name, param in visual_backbone.vision_model.encoder.layers[index].named_parameters():
            if "layer_norm" in name:
                param.requires_grad = True
    print(f"Unfreezing LayerNorm in CLIP encoder layers {target_layers}")

    model = CausalFINDClassifier(visual_backbone).to(config.device)
    criterion = nn.CrossEntropyLoss()
    main_params = (
        list(model.backbone.parameters())
        + list(model.causal_head.mask_mlp.parameters())
        + list(model.causal_head.classifier_h.parameters())
    )
    opt_main = optim.AdamW(
        filter(lambda param: param.requires_grad, main_params),
        lr=config.lr,
        weight_decay=1e-2,
    )
    opt_adv = optim.AdamW(model.causal_head.adversary_d.parameters(), lr=config.lr, weight_decay=1e-2)
    scheduler_main = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=config.epochs)
    scheduler_adv = optim.lr_scheduler.CosineAnnealingLR(opt_adv, T_max=config.epochs)
    return model, criterion, opt_main, opt_adv, scheduler_main, scheduler_adv


def compute_tpr_at_target_fpr(y_true, y_probs, target_fpr=0.05):
    if not y_true or not y_probs:
        return 0.0, None, None

    fpr, tpr, thresholds = roc_curve(np.asarray(y_true), np.asarray(y_probs))
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) == 0:
        return 0.0, None, None

    best_index = valid_indices[-1]
    return float(tpr[best_index]), float(thresholds[best_index]), float(fpr[best_index])


def compute_causal_clip_loss(model, imgs, labels, criterion, causal_mode):
    logits_h, logits_d, mask, logits_h_cf, logits_d_detached = model(imgs)
    loss_cls = criterion(logits_h, labels)
    loss_adv_d = criterion(logits_d_detached, labels)

    if causal_mode == "cls_only":
        return loss_cls, None, logits_h

    loss_mask = 0.01 * torch.mean(torch.abs(mask))
    if causal_mode == "partial":
        loss_total = loss_cls + loss_mask
        return loss_total, None, logits_h

    target_uninformative = torch.full_like(logits_d, 0.5)
    loss_adv_mask = criterion(logits_d, target_uninformative)
    log_prob_cf = F.log_softmax(logits_h_cf, dim=1)
    prob_h = F.softmax(logits_h, dim=1).detach()
    loss_inv = F.kl_div(log_prob_cf, prob_h, reduction="batchmean")
    loss_total = loss_cls + loss_inv + loss_adv_mask
    return loss_total, loss_adv_d, logits_h


def train_one_epoch(model, loader, opt_main, opt_adv, criterion, device, noise_std, causal_mode):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for imgs, lbls, _ in pbar:
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        real_mask = lbls == 0
        if real_mask.any():
            real_imgs = imgs[real_mask]
            noise = torch.randn_like(real_imgs) * noise_std
            noisy_imgs = torch.clamp(real_imgs + noise, 0.0, 1.0)
            noisy_lbls = torch.ones(noisy_imgs.size(0), dtype=torch.long, device=device)

            imgs = torch.cat([imgs, noisy_imgs], dim=0)
            lbls = torch.cat([lbls, noisy_lbls], dim=0)

            shuffle_idx = torch.randperm(imgs.size(0), device=device)
            imgs = imgs[shuffle_idx]
            lbls = lbls[shuffle_idx]

        imgs = frequency_domain_masking(imgs, mask_ratio=0.15)
        input_tensor = tensor_normalize(imgs)

        opt_main.zero_grad()
        if causal_mode == "full":
            opt_adv.zero_grad()
            _, loss_adv_d, _ = compute_causal_clip_loss(model, input_tensor, lbls, criterion, causal_mode)
            loss_adv_d.backward()

            for param in model.causal_head.adversary_d.parameters():
                param.requires_grad = False
            loss_total, _, logits_h = compute_causal_clip_loss(model, input_tensor, lbls, criterion, causal_mode)
            loss_total.backward()
            for param in model.causal_head.adversary_d.parameters():
                param.requires_grad = True
            opt_adv.step()
        else:
            loss_total, _, logits_h = compute_causal_clip_loss(model, input_tensor, lbls, criterion, causal_mode)
            loss_total.backward()
        opt_main.step()

        running_loss += loss_total.item()
        _, predicted = logits_h.max(1)
        total += lbls.size(0)
        correct += predicted.eq(lbls).sum().item()
        pbar.set_postfix({"Loss": f"{loss_total.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"})

    return running_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device, target_fpr=0.05):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_y_true = []
    all_y_probs = []

    with torch.no_grad():
        for imgs, lbls, _ in tqdm(loader, desc="Validating"):
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            imgs = tensor_normalize(imgs)
            logits_h = model(imgs)
            loss = criterion(logits_h, lbls)
            probs = torch.softmax(logits_h, dim=1)[:, 1]

            running_loss += loss.item()
            _, predicted = logits_h.max(1)
            total += lbls.size(0)
            correct += predicted.eq(lbls).sum().item()
            all_y_true.extend(lbls.cpu().tolist())
            all_y_probs.extend(probs.cpu().tolist())

    tpr_at_fpr, threshold_at_fpr, actual_fpr = compute_tpr_at_target_fpr(
        all_y_true,
        all_y_probs,
        target_fpr=target_fpr,
    )
    return (
        running_loss / len(loader),
        100.0 * correct / total,
        tpr_at_fpr,
        threshold_at_fpr,
        actual_fpr,
    )


def train_causal(config: CausalConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    train_loader, val_loader, ood_val_loader, held_out_generators = prepare_find_dataloaders_with_ood(
        fake_path=config.data_dir,
        max_ai_samples=config.max_ai_samples,
        batch_size=config.batch_size,
        ood_generator_fraction=config.ood_generator_fraction,
        ood_seed=config.ood_seed,
    )
    model, criterion, opt_main, opt_adv, scheduler_main, scheduler_adv = build_causal_model(config)
    print(f"Causal ablation mode: {config.causal_mode}")
    print(f"Held-out generators for OOD val: {', '.join(held_out_generators)}")

    csv_log_path = resolve_output_path(config.csv_log_path, output_dir=config.output_dir)
    write_header = not os.path.exists(csv_log_path)
    if write_header:
        with open(csv_log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "val_loss",
                    "val_acc",
                    "ood_val_loss",
                    "ood_val_acc",
                    "ood_val_tpr_at_5fpr",
                    "ood_val_threshold_at_5fpr",
                    "ood_val_actual_fpr",
                    "lr",
                    "causal_mode",
                ]
            )

    best_val_tpr_at_5fpr = -1.0
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        current_lr = opt_main.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            opt_main,
            opt_adv,
            criterion,
            device=config.device,
            noise_std=config.noise_std,
            causal_mode=config.causal_mode,
        )
        val_loss, val_acc, val_tpr_at_5fpr, val_threshold_at_5fpr, val_actual_fpr = validate(
            model,
            val_loader,
            criterion,
            device=config.device,
        )
        ood_val_loss, ood_val_acc, ood_val_tpr_at_5fpr, ood_val_threshold_at_5fpr, ood_val_actual_fpr = validate(
            model,
            ood_val_loader,
            criterion,
            device=config.device,
        )
        scheduler_main.step()
        if config.causal_mode == "full":
            scheduler_adv.step()
        print(f"Summary - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Summary - Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        if val_threshold_at_5fpr is not None:
            print(
                f"Summary - Val   TPR@5%FPR: {val_tpr_at_5fpr * 100:.2f}% "
                f"(thr={val_threshold_at_5fpr:.4f}, actual FPR={val_actual_fpr * 100:.2f}%)"
            )
        else:
            print("Summary - Val   TPR@5%FPR: unavailable")
        print(f"Summary - OOD Val Loss: {ood_val_loss:.4f} | OOD Val Acc: {ood_val_acc:.2f}%")
        if ood_val_threshold_at_5fpr is not None:
            print(
                f"Summary - OOD TPR@5%FPR: {ood_val_tpr_at_5fpr * 100:.2f}% "
                f"(thr={ood_val_threshold_at_5fpr:.4f}, actual FPR={ood_val_actual_fpr * 100:.2f}%)"
            )
        else:
            print("Summary - OOD TPR@5%FPR: unavailable")

        with open(csv_log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch + 1,
                    f"{train_loss:.4f}",
                    f"{train_acc:.2f}",
                    f"{val_loss:.4f}",
                    f"{val_acc:.2f}",
                    f"{ood_val_loss:.4f}",
                    f"{ood_val_acc:.2f}",
                    f"{ood_val_tpr_at_5fpr:.6f}",
                    "" if ood_val_threshold_at_5fpr is None else f"{ood_val_threshold_at_5fpr:.6f}",
                    "" if ood_val_actual_fpr is None else f"{ood_val_actual_fpr:.6f}",
                    f"{current_lr:.6f}",
                    config.causal_mode,
                ]
            )

        if ood_val_tpr_at_5fpr > best_val_tpr_at_5fpr:
            best_val_tpr_at_5fpr = ood_val_tpr_at_5fpr
            saved_path = save_checkpoint(
                model,
                config.model_save_path,
                output_dir=config.output_dir,
                record_path=config.checkpoint_record_path,
            )
            print(f"--> Best model saved (OOD TPR@5%FPR: {ood_val_tpr_at_5fpr * 100:.2f}%) at {saved_path}")

    print(f"\nTraining finished. Logs saved to: {csv_log_path}")
    return model


def load_causal_for_mnw(config: CausalConfig):
    checkpoint_path = resolve_checkpoint_path(
        config.model_save_path,
        output_dir=config.output_dir,
        record_path=config.checkpoint_record_path,
    )
    print(f"Using checkpoint: {checkpoint_path}")
    model, _, _, _, _, _ = build_causal_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    model.eval()
    return model
