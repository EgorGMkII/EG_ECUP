"""Training engine for Direct and Multi-Task GRU models with GPU acceleration and early stopping."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.sequential.dataset import OzonSequenceDataset
from src.sequential.losses import DirectMSELoss, MultiTaskHurdleLoss
from src.sequential.models import DirectGRUModel, MultiTaskGRUModel

MODELS_DIR = Path("models/sequential")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_direct_gru(
    model: DirectGRUModel,
    train_dataset: OzonSequenceDataset,
    val_dataset: Optional[OzonSequenceDataset] = None,
    epochs: int = 15,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    clip_grad: float = 1.0,
    patience: int = 4,
    checkpoint_path: Path = MODELS_DIR / "direct_gru_best.pt",
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Trains Direct GRU Model on GPU with AdamW, Cosine Scheduler, and early stopping."""
    device = get_device()
    model = model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, pin_memory=True, num_workers=0) if val_dataset else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = DirectMSELoss()

    history = {"train_loss": [], "val_loss": [], "val_rmsle": []}
    best_val_rmsle = float("inf")
    patience_counter = 0

    if verbose:
        print(f"[*] Starting Direct GRU Training on {device} ({len(train_dataset):,} samples, batch_size={batch_size})...")

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y_log, _, _ in train_loader:
            x, y_log = x.to(device), y_log.to(device)
            optimizer.zero_grad()
            pred, _ = model(x)
            loss = criterion(pred, y_log)
            loss.backward()

            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(1, n_batches)
        history["train_loss"].append(train_loss)

        # Validation step
        val_loss, val_rmsle = 0.0, 0.0
        if val_loader:
            model.eval()
            val_preds_list, val_targets_list = [], []
            with torch.no_grad():
                for x, y_log, _, y_raw in val_loader:
                    x, y_log = x.to(device), y_log.to(device)
                    pred, _ = model(x)
                    v_loss = criterion(pred, y_log)
                    val_loss += v_loss.item()

                    pred_clipped = torch.clamp(pred, min=0.0).cpu().numpy()
                    y_log_np = y_log.cpu().numpy()
                    val_preds_list.append(pred_clipped)
                    val_targets_list.append(y_log_np)

            val_loss /= max(1, len(val_loader))
            all_preds = np.concatenate(val_preds_list)
            all_targets = np.concatenate(val_targets_list)
            val_rmsle = float(np.sqrt(np.mean((all_preds - all_targets) ** 2)))

            history["val_loss"].append(val_loss)
            history["val_rmsle"].append(val_rmsle)

            if verbose:
                print(f"  Epoch [{epoch+1:02d}/{epochs:02d}] ({time.time()-t0:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSLE: {val_rmsle:.4f}")

            if val_rmsle < best_val_rmsle:
                best_val_rmsle = val_rmsle
                patience_counter = 0
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"[*] Early stopping triggered at epoch {epoch+1} (Best Val RMSLE: {best_val_rmsle:.4f})")
                    break
        else:
            if verbose:
                print(f"  Epoch [{epoch+1:02d}/{epochs:02d}] ({time.time()-t0:.1f}s) | Train Loss: {train_loss:.4f}")

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    return history


def train_multitask_gru(
    model: MultiTaskGRUModel,
    train_dataset: OzonSequenceDataset,
    val_dataset: Optional[OzonSequenceDataset] = None,
    epochs: int = 15,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_cls: float = 0.5,
    lambda_reg: float = 1.0,
    alpha: float = 1.1,
    clip_grad: float = 1.0,
    patience: int = 4,
    checkpoint_path: Path = MODELS_DIR / "multitask_gru_best.pt",
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Trains Multi-Task Hurdle GRU (Classification + Conditional Regression)."""
    device = get_device()
    model = model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, pin_memory=True, num_workers=0) if val_dataset else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = MultiTaskHurdleLoss(lambda_cls=lambda_cls, lambda_reg=lambda_reg)

    history = {"train_loss": [], "val_loss": [], "val_rmsle": [], "val_brier": []}
    best_val_rmsle = float("inf")
    patience_counter = 0

    if verbose:
        print(f"[*] Starting Multi-Task GRU Training on {device} (lambda_cls={lambda_cls}, lambda_reg={lambda_reg})...")

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y_log, is_buyer, _ in train_loader:
            x, y_log, is_buyer = x.to(device), y_log.to(device), is_buyer.to(device)
            optimizer.zero_grad()
            p_logits, z_cond, _ = model(x)
            loss, _ = criterion(p_logits, z_cond, is_buyer, y_log)
            loss.backward()

            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(1, n_batches)
        history["train_loss"].append(train_loss)

        # Validation step
        if val_loader:
            model.eval()
            p_list, z_list, target_list = [], [], []
            val_loss = 0.0
            with torch.no_grad():
                for x, y_log, is_buyer, y_raw in val_loader:
                    x, y_log, is_buyer = x.to(device), y_log.to(device), is_buyer.to(device)
                    p_logits, z_cond, _ = model(x)
                    v_loss, _ = criterion(p_logits, z_cond, is_buyer, y_log)
                    val_loss += v_loss.item()

                    p_buy = torch.sigmoid(p_logits).cpu().numpy()
                    z_c = torch.clamp(z_cond, min=0.0).cpu().numpy()
                    y_log_np = y_log.cpu().numpy()

                    p_list.append(p_buy)
                    z_list.append(z_c)
                    target_list.append(y_log_np)

            val_loss /= max(1, len(val_loader))
            all_p = np.concatenate(p_list)
            all_z = np.concatenate(z_list)
            all_y = np.concatenate(target_list)

            # Hurdle composite prediction
            z_pred_hurdle = np.power(all_p, alpha) * all_z
            val_rmsle = float(np.sqrt(np.mean((z_pred_hurdle - all_y) ** 2)))
            val_brier = float(np.mean((all_p - (all_y > 0).astype(float)) ** 2))

            history["val_loss"].append(val_loss)
            history["val_rmsle"].append(val_rmsle)
            history["val_brier"].append(val_brier)

            if verbose:
                print(f"  Epoch [{epoch+1:02d}/{epochs:02d}] ({time.time()-t0:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSLE: {val_rmsle:.4f} | Brier: {val_brier:.4f}")

            if val_rmsle < best_val_rmsle:
                best_val_rmsle = val_rmsle
                patience_counter = 0
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"[*] Early stopping triggered at epoch {epoch+1} (Best Val RMSLE: {best_val_rmsle:.4f})")
                    break

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    return history
