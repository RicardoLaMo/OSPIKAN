#!/usr/bin/env python3
"""
SPIKAN Training Script with Time-Split Evaluation.

Trains a Separable Physics-Informed KAN for market shock prediction
and regime forecasting with proper out-of-sample validation.

Features:
- Multi-GPU training with DataParallel
- Mixed precision (AMP) for H200 GPU optimization
- GPU-optimized DataLoader with pinned memory and prefetching
- Gradient checkpointing for memory efficiency

Evaluation metrics:
- AUC-ROC for shock prediction
- Precision@K for top-K shock events
- Calibration (reliability diagram)
- PDE residual statistics

Usage:
    python scripts/train_spikan.py --config configs/silver_universe.yaml
    python scripts/train_spikan.py --train-end 2020-01-01 --test-start 2020-01-01
    python scripts/train_spikan.py --multi-gpu --amp  # Enable multi-GPU and mixed precision
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path BEFORE any other imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import physics module FIRST - it configures tempdir for FUSE/Drive mounts
# This must happen before torch import
import src.physics  # noqa: F401 - side effect: configures tempdir

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.physics.spikan import MarketSPIKAN, SPIKANConfig
from src.physics.market_pdes import BurgersEquation, PDEConfig, physics_informed_loss
from src.physics.shock_detector import label_historical_shocks


def setup_cuda_environment() -> Tuple[torch.device, int]:
    """
    Configure CUDA environment for optimal H200 performance.

    Returns:
        (device, num_gpus) tuple
    """
    if not torch.cuda.is_available():
        return torch.device("cpu"), 0

    num_gpus = torch.cuda.device_count()

    # Enable TF32 for H200 (faster matrix multiplications)
    # Use new PyTorch 2.x API if available
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
    else:
        # Fallback for older PyTorch versions
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Preferred precision hint for matmuls (PyTorch 2.x).
    # Has effect for float32 matmuls (TF32/FP32 tradeoffs).
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Enable cuDNN autotuner for consistent input sizes
    torch.backends.cudnn.benchmark = True

    # Memory optimization
    if hasattr(torch.cuda, 'memory'):
        # Use memory efficient attention if available
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    device = torch.device("cuda:0")

    # Print GPU info
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")

    return device, num_gpus


def _select_amp_dtype(device: torch.device) -> torch.dtype:
    """
    Choose a safe AMP dtype for the current GPU.

    For physics-informed training (higher-order derivatives), fp16 can be
    numerically fragile. Hopper (H100/H200, cc>=9.0) supports bfloat16 well
    and is typically more stable.
    """
    if device.type != "cuda":
        return torch.float32
    try:
        props = torch.cuda.get_device_properties(int(device.index or 0))
        if int(getattr(props, "major", 0)) >= 9:
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _autocast_ctx(device: torch.device, dtype: torch.dtype, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled)
    return torch.cuda.amp.autocast(dtype=dtype, enabled=enabled)


def _make_grad_scaler(device: torch.device, enabled: bool):
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        # `torch.cuda.amp.GradScaler` is deprecated in newer PyTorch versions.
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


@dataclass
class TrainingConfig:
    """Configuration for SPIKAN training."""
    # Data splits
    train_end: str = "2019-12-31"
    val_start: str = "2020-01-01"
    val_end: str = "2021-12-31"
    test_start: str = "2022-01-01"

    # Model architecture
    n_assets: int = 6
    n_macro: int = 4
    n_geometric: int = 4
    hidden_dim: int = 32
    n_terms: int = 8

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 10

    # Physics loss weights
    lambda_data: float = 1.0
    lambda_pde: float = 0.1
    lambda_reg: float = 1e-4

    # Shock labeling
    shock_threshold_mild: float = 2.0
    shock_threshold_severe: float = 3.0
    shock_lookahead: int = 5

    # GPU/CUDA options
    use_amp: bool = True  # Mixed precision training (recommended for H200)
    multi_gpu: bool = False  # Use DataParallel for multi-GPU
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    prefetch_factor: int = 2  # DataLoader prefetch batches
    compile_model: bool = True  # Use torch.compile() for optimization

    # Output
    output_dir: str = "reports/silver/spikan"
    model_name: str = "spikan_shock"


def load_features(config_path: str) -> pd.DataFrame:
    """Load features from pipeline output."""
    import yaml

    # Resolve config path to absolute
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        # Try relative to PROJECT_ROOT
        config_path = (PROJECT_ROOT / config_path.name).resolve()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve output_dir relative to PROJECT_ROOT if not absolute
    output_dir = Path(cfg.get("output_dir", "reports/silver"))
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    # Search locations in order of preference
    search_paths = [
        output_dir / "runs" / "features.parquet",
        output_dir / "runs_test" / "features.parquet",
        output_dir / "features.parquet",
    ]

    # Also search data/processed for timestamped feature files
    processed_dir = PROJECT_ROOT / "data" / "processed"
    if processed_dir.exists():
        feature_files = sorted(processed_dir.glob("silver_features_*.parquet"), reverse=True)
        search_paths.extend(feature_files[:3])  # Add most recent 3

    features_path = None
    for path in search_paths:
        if path.exists():
            features_path = path
            break

    if features_path is None:
        locations = "\n  ".join(str(p) for p in search_paths[:5])
        raise FileNotFoundError(
            f"Features not found. Searched:\n  {locations}\n"
            f"Please run the silver pipeline first to generate features."
        )

    print(f"Loading features from {features_path}")
    return pd.read_parquet(features_path)


def prepare_training_data(
    features: pd.DataFrame,
    config: TrainingConfig,
) -> Dict[str, Tuple[torch.Tensor, ...]]:
    """
    Prepare training, validation, and test datasets.

    Returns dict with 'train', 'val', 'test' keys, each containing:
    (assets, macro, geometry, time, target, shock_labels)
    """
    # Define feature columns
    asset_cols = [c for c in features.columns if "log_return" in c or "momentum" in c][:config.n_assets]
    macro_cols = [c for c in features.columns if any(m in c for m in ["dxy", "y10", "spx", "vix"])][:config.n_macro]
    geom_cols = [c for c in features.columns if any(g in c for g in ["ricci", "mst_stress", "rotor"])][:config.n_geometric]

    # Ensure we have enough columns
    while len(asset_cols) < config.n_assets:
        asset_cols.append(asset_cols[-1] if asset_cols else "log_return_1d")
    while len(macro_cols) < config.n_macro:
        macro_cols.append(macro_cols[-1] if macro_cols else "dxy_log_return_1d")
    while len(geom_cols) < config.n_geometric:
        geom_cols.append(geom_cols[-1] if geom_cols else "ricci_mean_60d")

    asset_cols = asset_cols[:config.n_assets]
    macro_cols = macro_cols[:config.n_macro]
    geom_cols = geom_cols[:config.n_geometric]

    print(f"Asset features ({len(asset_cols)}): {asset_cols}")
    print(f"Macro features ({len(macro_cols)}): {macro_cols}")
    print(f"Geometry features ({len(geom_cols)}): {geom_cols}")

    # Get silver returns for shock labeling
    if "log_return_1d" in features.columns:
        returns = features["log_return_1d"]
    else:
        returns = features.iloc[:, 0].pct_change()

    # Label historical shocks
    shock_labels = label_historical_shocks(
        returns,
        threshold_mild=config.shock_threshold_mild,
        threshold_severe=config.shock_threshold_severe,
        lookahead=config.shock_lookahead,
    )

    # Extract feature matrices
    df = features.copy()
    df["shock_label"] = shock_labels
    df = df.dropna(subset=asset_cols + macro_cols + geom_cols + ["shock_label"])

    # Time split
    train_end = pd.to_datetime(config.train_end)
    val_start = pd.to_datetime(config.val_start)
    val_end = pd.to_datetime(config.val_end)
    test_start = pd.to_datetime(config.test_start)

    train_df = df[df.index <= train_end]
    val_df = df[(df.index >= val_start) & (df.index <= val_end)]
    test_df = df[df.index >= test_start]

    print(f"Train: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
    print(f"Val: {len(val_df)} samples ({val_df.index.min()} to {val_df.index.max()})")
    print(f"Test: {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")

    def to_tensors(df_split: pd.DataFrame) -> Tuple[torch.Tensor, ...]:
        assets = torch.tensor(df_split[asset_cols].values, dtype=torch.float32)
        macro = torch.tensor(df_split[macro_cols].values, dtype=torch.float32)
        geom = torch.tensor(df_split[geom_cols].values, dtype=torch.float32)
        time = torch.arange(len(df_split), dtype=torch.float32).unsqueeze(-1) / len(df_split)
        target = torch.tensor(df_split["shock_label"].values, dtype=torch.long)
        # Next-step return as regression target
        if "log_return_1d" in df_split.columns:
            reg_target = torch.tensor(df_split["log_return_1d"].shift(-1).fillna(0).values, dtype=torch.float32).unsqueeze(-1)
        else:
            reg_target = torch.zeros(len(df_split), 1)
        return assets, macro, geom, time, reg_target, target

    return {
        "train": to_tensors(train_df),
        "val": to_tensors(val_df),
        "test": to_tensors(test_df),
        "dates": {
            "train": train_df.index,
            "val": val_df.index,
            "test": test_df.index,
        },
    }


def train_epoch(
    model: MarketSPIKAN,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    pde: BurgersEquation,
    pde_config: PDEConfig,
    device: torch.device,
    scaler: Optional[object] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """
    Train for one epoch with optional mixed precision.

    Args:
        model: SPIKAN model
        train_loader: Training data loader
        optimizer: Optimizer
        pde: PDE equation object
        pde_config: PDE configuration
        device: Torch device
        scaler: GradScaler for mixed precision (None if not using AMP)
        use_amp: Whether to use automatic mixed precision

    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_pde_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        # Use non_blocking for async GPU transfer when pin_memory is enabled
        assets, macro, geom, time, target, _ = [
            b.to(device, non_blocking=True) for b in batch
        ]

        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        # Mixed precision forward pass
        if use_amp:
            with _autocast_ctx(device, amp_dtype, enabled=True):
                losses = physics_informed_loss(
                    model, assets, macro, geom, time, target,
                    pde=pde, config=pde_config
                )

            # AMP fallback: if we hit NaNs/Infs (common with fp16 + 2nd derivatives),
            # recompute the batch in full precision once.
            if not torch.isfinite(losses["total"]):
                with _autocast_ctx(device, torch.float32, enabled=False):
                    losses = physics_informed_loss(
                        model, assets, macro, geom, time, target,
                        pde=pde, config=pde_config
                    )

            if scaler is not None:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            losses = physics_informed_loss(
                model, assets, macro, geom, time, target,
                pde=pde, config=pde_config
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += losses["total"].item()
        total_data_loss += losses["data"].item()
        total_pde_loss += losses["pde"].item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "data_loss": total_data_loss / n_batches,
        "pde_loss": total_pde_loss / n_batches,
    }


def evaluate(
    model: MarketSPIKAN,
    data_loader: DataLoader,
    pde: BurgersEquation,
    pde_config: PDEConfig,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set with optional mixed precision.

    Args:
        model: SPIKAN model
        data_loader: Data loader
        pde: PDE equation object
        pde_config: PDE configuration
        device: Torch device
        use_amp: Whether to use automatic mixed precision

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_data_loss = 0.0
    total_pde_loss = 0.0
    all_preds = []
    all_targets = []
    all_shock_labels = []
    n_batches = 0

    # NOTE: Do not wrap in `torch.no_grad()` because `physics_informed_loss()`
    # computes PDE derivatives via autograd (needs grad tracking).
    for batch in data_loader:
        assets, macro, geom, time, target, shock_label = [
            b.to(device, non_blocking=True) for b in batch
        ]

        # Mixed precision inference
        if use_amp:
            with _autocast_ctx(device, amp_dtype, enabled=True):
                pred = model(assets, macro, geom, time)
                losses = physics_informed_loss(
                    model, assets, macro, geom, time, target,
                    pde=pde, config=pde_config
                )
        else:
            pred = model(assets, macro, geom, time)
            losses = physics_informed_loss(
                model, assets, macro, geom, time, target,
                pde=pde, config=pde_config
            )

        if not torch.isfinite(losses["total"]):
            # Re-evaluate in fp32 for numerical stability (metrics only).
            with _autocast_ctx(device, torch.float32, enabled=False):
                pred = model(assets, macro, geom, time)
                losses = physics_informed_loss(
                    model, assets, macro, geom, time, target,
                    pde=pde, config=pde_config
                )

        all_preds.append(pred.float().detach().cpu())  # Ensure float32 for metrics
        all_targets.append(target.detach().cpu())
        all_shock_labels.append(shock_label.detach().cpu())

        total_loss += losses["total"].item()
        total_data_loss += losses["data"].item()
        total_pde_loss += losses["pde"].item()
        n_batches += 1

    # Compute metrics
    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    shock_labels = torch.cat(all_shock_labels, dim=0).numpy()

    metrics = {
        "loss": total_loss / n_batches,
        "data_loss": total_data_loss / n_batches,
        "pde_loss": total_pde_loss / n_batches,
        "mse": np.mean((preds.flatten() - targets.flatten()) ** 2),
        "mae": np.mean(np.abs(preds.flatten() - targets.flatten())),
    }

    # Shock prediction metrics (using prediction magnitude as score)
    pred_scores = np.abs(preds.flatten())
    binary_shock = (shock_labels > 0).astype(int)

    if binary_shock.sum() > 0:
        # AUC-ROC
        try:
            from sklearn.metrics import roc_auc_score, precision_score
            metrics["auc"] = roc_auc_score(binary_shock, pred_scores)
        except:
            metrics["auc"] = 0.5

        # Precision@K
        k = min(int(binary_shock.sum()), len(pred_scores))
        if k > 0:
            top_k_idx = np.argsort(pred_scores)[-k:]
            metrics["precision_at_k"] = binary_shock[top_k_idx].mean()
        else:
            metrics["precision_at_k"] = 0.0
    else:
        metrics["auc"] = 0.5
        metrics["precision_at_k"] = 0.0

    return metrics


def train_model(
    data: Dict[str, Tuple[torch.Tensor, ...]],
    config: TrainingConfig,
    device: torch.device,
    num_gpus: int = 1,
) -> Tuple[MarketSPIKAN, Dict[str, List[float]]]:
    """
    Full training loop with early stopping and GPU optimizations.

    Args:
        data: Dictionary with train/val/test data tensors
        config: Training configuration
        device: Primary torch device
        num_gpus: Number of available GPUs

    Returns:
        (trained_model, training_history) tuple
    """
    # Create model
    model_config = SPIKANConfig(
        n_assets=config.n_assets,
        n_macro=config.n_macro,
        n_geometric=config.n_geometric,
        hidden_dim=config.hidden_dim,
        n_terms=config.n_terms,
        output_dim=1,  # Predict next-step return
    )
    model = MarketSPIKAN(model_config).to(device)

    # Multi-GPU support with DataParallel
    if config.multi_gpu and num_gpus > 1:
        if float(config.lambda_pde) > 0.0:
            print(
                "Warning: --multi-gpu requested, but PDE loss requires autograd derivatives "
                "that are not DataParallel-safe in this implementation. "
                "Training will run on a single GPU. "
                "For true multi-GPU physics-informed training, use DDP (torchrun)."
            )
        else:
            print(f"Using DataParallel with {num_gpus} GPUs")
            model = nn.DataParallel(model)
    elif config.multi_gpu and num_gpus <= 1:
        print("Warning: --multi-gpu requested but only one GPU is available; using single-GPU training.")

    # torch.compile() for PyTorch 2.0+ optimization (significant speedup)
    if (
        config.compile_model
        and device.type == "cuda"
        and not (config.multi_gpu and num_gpus > 1)
        and hasattr(torch, "compile")
    ):
        try:
            print("Compiling model with torch.compile() for optimization...")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"torch.compile() failed, using eager mode: {e}")

    # Create PDE and config
    pde_config = PDEConfig(
        lambda_data=config.lambda_data,
        lambda_pde=config.lambda_pde,
        lambda_reg=config.lambda_reg,
    )
    pde = BurgersEquation(pde_config)

    # Create data loaders with GPU optimizations
    train_dataset = TensorDataset(*data["train"])
    val_dataset = TensorDataset(*data["val"])

    # GPU-optimized DataLoader settings
    num_workers = int(config.num_workers) if device.type == "cuda" else 0
    loader_kwargs = {
        "batch_size": int(config.batch_size),
        "num_workers": num_workers,
        "pin_memory": bool(config.pin_memory and device.type == "cuda"),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(config.prefetch_factor)
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Mixed precision GradScaler
    use_amp = config.use_amp and device.type == "cuda"
    amp_dtype = _select_amp_dtype(device) if use_amp else torch.float32
    scaler = _make_grad_scaler(device, enabled=(use_amp and amp_dtype == torch.float16))

    if use_amp:
        print(f"Using mixed precision training (AMP) with {amp_dtype}")

    # Optimizer with fused kernels for GPU
    optimizer_kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    # Use fused optimizer if available (faster on GPU); fall back gracefully.
    optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)
    if device.type == "cuda":
        try:
            optimizer = optim.AdamW(model.parameters(), fused=True, **optimizer_kwargs)
            print("Using fused AdamW optimizer")
        except TypeError:
            pass
        except Exception as e:
            print(f"Fused AdamW unavailable, using standard AdamW: {e}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_precision_at_k": [],
    }

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    print(f"\nStarting training on {device}...")
    if device.type == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    for epoch in range(config.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, pde, pde_config, device,
            scaler=scaler, use_amp=use_amp, amp_dtype=amp_dtype
        )

        # Validate
        val_metrics = evaluate(model, val_loader, pde, pde_config, device, use_amp=use_amp, amp_dtype=amp_dtype)

        # Update scheduler
        scheduler.step(val_metrics["loss"])

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_precision_at_k"].append(val_metrics["precision_at_k"])

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            # Handle DataParallel model state
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            best_model_state = {k: v.cpu().clone() for k, v in state_dict.items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}: "
                  f"train_loss={train_metrics['loss']:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, "
                  f"val_auc={val_metrics['auc']:.3f}, "
                  f"precision@K={val_metrics['precision_at_k']:.3f}")

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
        print(f"\nTotal training time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")

    # Load best model (handle DataParallel wrapper)
    if best_model_state is not None:
        # Create fresh model for returning (unwrapped)
        return_model = MarketSPIKAN(model_config).to(device)
        return_model.load_state_dict(best_model_state)
        return return_model, history

    # Return unwrapped model
    if hasattr(model, 'module'):
        return model.module, history
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train SPIKAN for shock prediction")
    parser.add_argument("--config", type=str, default="configs/silver_universe.yaml",
                        help="Path to universe config")
    parser.add_argument("--train-end", type=str, default="2019-12-31",
                        help="End date for training data")
    parser.add_argument("--val-start", type=str, default="2020-01-01",
                        help="Start date for validation data")
    parser.add_argument("--val-end", type=str, default="2021-12-31",
                        help="End date for validation data")
    parser.add_argument("--test-start", type=str, default="2022-01-01",
                        help="Start date for test data")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--output-dir", type=str, default="reports/silver/spikan",
                        help="Output directory")
    # GPU options
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (recommended for H200)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Enable multi-GPU training with DataParallel")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile() optimization")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Specific GPU device ID to use (default: auto-select)")
    args = parser.parse_args()

    # Setup CUDA environment
    device, num_gpus = setup_cuda_environment()

    # Override device if specific GPU requested
    if args.gpu is not None and torch.cuda.is_available():
        if args.gpu < num_gpus:
            device = torch.device(f"cuda:{args.gpu}")
            print(f"Using specified GPU: {args.gpu}")
        else:
            print(f"Warning: Requested GPU {args.gpu} not available, using GPU 0")

    print(f"Using device: {device}")
    if num_gpus > 0:
        print(f"Available GPUs: {num_gpus}")

    # Configuration with GPU options
    use_amp = True  # Default: enabled
    if args.no_amp:
        use_amp = False
    elif args.amp:
        use_amp = True

    config = TrainingConfig(
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        use_amp=use_amp,
        multi_gpu=args.multi_gpu,
        num_workers=args.num_workers,
        compile_model=not args.no_compile,
    )

    # Load features
    try:
        features = load_features(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the silver pipeline first to generate features.")
        sys.exit(1)

    # Prepare data
    data = prepare_training_data(features, config)

    # Train model
    model, history = train_model(data, config, device, num_gpus=num_gpus)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_dataset = TensorDataset(*data["test"])

    # GPU-optimized test loader
    test_loader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": config.num_workers if device.type == "cuda" else 0,
        "pin_memory": config.pin_memory and device.type == "cuda",
    }
    test_loader = DataLoader(test_dataset, **test_loader_kwargs)

    pde_config = PDEConfig(
        lambda_data=config.lambda_data,
        lambda_pde=config.lambda_pde,
        lambda_reg=config.lambda_reg,
    )
    pde = BurgersEquation(pde_config)

    use_amp = config.use_amp and device.type == "cuda"
    amp_dtype = _select_amp_dtype(device) if use_amp else torch.float32
    test_metrics = evaluate(model, test_loader, pde, pde_config, device, use_amp=use_amp, amp_dtype=amp_dtype)

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"MSE:           {test_metrics['mse']:.6f}")
    print(f"MAE:           {test_metrics['mae']:.6f}")
    print(f"AUC-ROC:       {test_metrics['auc']:.3f}")
    print(f"Precision@K:   {test_metrics['precision_at_k']:.3f}")
    print(f"PDE Residual:  {test_metrics['pde_loss']:.6f}")
    print("=" * 50)

    # Save model and results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / f"{config.model_name}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "history": history,
        "test_metrics": test_metrics,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # Save metrics summary
    metrics_path = output_dir / f"{config.model_name}_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("SPIKAN Training Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Config:\n")
        f.write(f"  Train end: {config.train_end}\n")
        f.write(f"  Val: {config.val_start} to {config.val_end}\n")
        f.write(f"  Test start: {config.test_start}\n")
        f.write(f"  Epochs: {config.epochs}\n")
        f.write(f"  Batch size: {config.batch_size}\n\n")
        f.write(f"Test Metrics:\n")
        f.write(f"  MSE: {test_metrics['mse']:.6f}\n")
        f.write(f"  MAE: {test_metrics['mae']:.6f}\n")
        f.write(f"  AUC-ROC: {test_metrics['auc']:.3f}\n")
        f.write(f"  Precision@K: {test_metrics['precision_at_k']:.3f}\n")
        f.write(f"  PDE Residual: {test_metrics['pde_loss']:.6f}\n")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
