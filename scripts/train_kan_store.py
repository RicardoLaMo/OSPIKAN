#!/usr/bin/env python
"""
Training script for KAN Knowledge Store.

Generates synthetic training data and trains 3 KAN models:
  - VolSurfaceKAN: regime + moneyness + time -> implied volatility
  - CovarianceKAN: regime -> 6x6 PSD covariance matrix (Cholesky)
  - TransitionKAN: regime + horizon -> regime transition probabilities

Usage:
    python scripts/train_kan_store.py --epochs 100 --batch-size 32
    python scripts/train_kan_store.py --synthetic-only --checkpoint reports/options/kan_store/final
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.options.kan_store.config import KANStoreConfig
from src.options.kan_store.store import KANKnowledgeStore


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_synthetic_vol_surface_data(
    num_samples: int, regime_specs: dict, device: str = "cpu"
):
    """
    Generate synthetic vol surface training data.

    Input: (regime features [8], log_moneyness [1], time_fraction [1])
    Output: implied volatility [1]
    """
    regime_features = []
    log_moneyness_vals = []
    time_fractions = []
    vols = []

    regime_order = ["STABLE", "TRANSITION", "STRESS", "RECOVERY"]

    for _ in range(num_samples):
        # Random regime selection
        regime_idx = np.random.randint(0, 4)
        regime_name = regime_order[regime_idx]

        # Generate regime feature vector
        # Simple representation: one-hot-like encoding of regime with added noise
        regime_feat = np.zeros(8)
        regime_feat[regime_idx] = 1.0
        regime_feat += np.random.normal(0, 0.1, 8)  # Add noise

        # Random moneyness: log(S/K) in [-0.2, 0.2]
        log_m = np.random.uniform(-0.2, 0.2)

        # Random time: 0.02 to 2.0 years (7 to 730 days)
        T = np.random.uniform(0.02, 2.0)

        # Base vol for regime
        base_vol = regime_specs[regime_name]["base_vol"]
        smile_steepness = regime_specs[regime_name]["smile_steepness"]

        # Vol surface model: base_vol + smile effect
        smile_effect = smile_steepness * abs(log_m) ** 1.5
        vol = base_vol + smile_effect

        # Add some noise
        vol += np.random.normal(0, 0.01)
        vol = np.clip(vol, 0.05, 1.0)  # Reasonable range

        regime_features.append(regime_feat)
        log_moneyness_vals.append([log_m])
        time_fractions.append([T])
        vols.append([vol])

    # Stack and convert to tensors
    X_regime = torch.tensor(np.array(regime_features), dtype=torch.float32)
    X_moneyness = torch.tensor(np.array(log_moneyness_vals), dtype=torch.float32)
    X_time = torch.tensor(np.array(time_fractions), dtype=torch.float32)
    y = torch.tensor(np.array(vols), dtype=torch.float32)

    X = torch.cat([X_regime, X_moneyness, X_time], dim=1)  # [N, 10]

    return X.to(device), y.to(device)


def generate_synthetic_covariance_data(
    num_samples: int, device: str = "cpu"
):
    """
    Generate synthetic covariance training data.

    Input: regime features [8]
    Output: Cholesky lower triangle of 6x6 cov [21] (upper triangle + diagonal for sym)
    """
    regime_features = []
    cholesky_vecs = []

    regime_order = ["STABLE", "TRANSITION", "STRESS", "RECOVERY"]

    for _ in range(num_samples):
        # Random regime
        regime_idx = np.random.randint(0, 4)
        regime_name = regime_order[regime_idx]

        # Regime feature vector
        regime_feat = np.zeros(8)
        regime_feat[regime_idx] = 1.0
        regime_feat += np.random.normal(0, 0.1, 8)

        # Generate regime-characteristic correlation structure
        # In STABLE: high correlation (metals correlated)
        # In STRESS: collapse (decorrelation or negative)
        # In TRANSITION/RECOVERY: intermediate

        if regime_name == "STABLE":
            base_corr = 0.7
            diag_vols = np.array([0.15, 0.16, 0.20, 0.25, 0.30, 0.08])
        elif regime_name == "STRESS":
            base_corr = 0.3
            diag_vols = np.array([0.40, 0.42, 0.50, 0.35, 0.40, 0.25])
        elif regime_name == "TRANSITION":
            base_corr = 0.5
            diag_vols = np.array([0.25, 0.27, 0.35, 0.30, 0.35, 0.15])
        else:  # RECOVERY
            base_corr = 0.6
            diag_vols = np.array([0.20, 0.22, 0.28, 0.28, 0.32, 0.10])

        # Build correlation matrix
        rho = np.ones((6, 6)) * base_corr
        np.fill_diagonal(rho, 1.0)

        # Add random perturbation
        rho += np.random.normal(0, 0.05, (6, 6))
        rho = (rho + rho.T) / 2  # Symmetrize
        np.fill_diagonal(rho, 1.0)

        # Covariance: diag(vols) @ rho @ diag(vols)
        D = np.diag(diag_vols)
        cov = D @ rho @ D

        # Ensure PSD by adding small regularization
        eigvals = np.linalg.eigvals(cov)
        if np.any(eigvals < 1e-6):
            cov += np.eye(6) * (1e-4)

        # Cholesky decomposition
        L = np.linalg.cholesky(cov)

        # Vectorize: flatten to [21] (upper triangle + diag)
        chol_vec = L[np.triu_indices(6)]  # Upper triangle

        regime_features.append(regime_feat)
        cholesky_vecs.append(chol_vec)

    X = torch.tensor(np.array(regime_features), dtype=torch.float32)
    y = torch.tensor(np.array(cholesky_vecs), dtype=torch.float32)

    return X.to(device), y.to(device)


def generate_synthetic_transition_data(
    num_samples: int, device: str = "cpu"
):
    """
    Generate synthetic transition matrix training data.

    Input: regime features [8] + horizon [1]
    Output: softmax probabilities over 4 regimes [4]
    """
    regime_features = []
    horizons = []
    probs = []

    regime_order = ["STABLE", "TRANSITION", "STRESS", "RECOVERY"]

    for _ in range(num_samples):
        # Current regime
        from_regime_idx = np.random.randint(0, 4)
        from_regime = regime_order[from_regime_idx]

        # Regime feature vector
        regime_feat = np.zeros(8)
        regime_feat[from_regime_idx] = 1.0
        regime_feat += np.random.normal(0, 0.1, 8)

        # Horizon in years
        horizon = np.random.uniform(0.01, 2.0)

        # Regime transition probabilities (simple model)
        # Probability of staying in same regime decreases with horizon
        # Probability of other regimes increases
        diag_prob = np.exp(-horizon / 0.5)  # Decay with horizon
        off_diag = (1.0 - diag_prob) / 3.0

        # Base transition matrix
        trans_matrix = np.ones((4, 4)) * off_diag
        np.fill_diagonal(trans_matrix, diag_prob)

        # Regime-specific transitions
        if from_regime == "STRESS":
            # More likely to recover
            trans_matrix[from_regime_idx, regime_order.index("RECOVERY")] *= 1.5
        elif from_regime == "RECOVERY":
            # More likely to go back to stable
            trans_matrix[from_regime_idx, regime_order.index("STABLE")] *= 1.3

        # Normalize to probabilities
        trans_probs = trans_matrix[from_regime_idx]
        trans_probs /= trans_probs.sum()

        regime_features.append(regime_feat)
        horizons.append([horizon])
        probs.append(trans_probs)

    X_regime = torch.tensor(np.array(regime_features), dtype=torch.float32)
    X_horizon = torch.tensor(np.array(horizons), dtype=torch.float32)
    X = torch.cat([X_regime, X_horizon], dim=1)  # [N, 9]
    y = torch.tensor(np.array(probs), dtype=torch.float32)

    return X.to(device), y.to(device)


def train_kan_model(model, train_loader, device, epochs, learning_rate, model_name):
    """Train a single KAN model."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.to(device)
    model.train()

    losses = []
    pbar = tqdm(range(epochs), desc=f"Training {model_name}")

    for epoch in pbar:
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # Handle different input formats based on model type
            if model_name == "VolSurfaceKAN":
                # regime[8] + log_moneyness[1] + time[1] = 10
                regime = X_batch[:, :8]
                log_moneyness = X_batch[:, 8:9]
                time_to_expiry = X_batch[:, 9:10]
                y_pred = model(regime, log_moneyness, time_to_expiry)
            elif model_name == "CovarianceKAN":
                # regime[8]
                y_pred = model(X_batch)  # Output: (batch, 6, 6) covariance
                # Reconstruct target covariance from Cholesky factors
                batch_size = y_batch.shape[0]
                L_target = torch.zeros(batch_size, 6, 6, device=device)
                idx = 0
                for i in range(6):
                    for j in range(i + 1):
                        L_target[:, i, j] = y_batch[:, idx]
                        idx += 1
                # Reconstruct covariance: Cov = L @ L.T
                y_batch = torch.bmm(L_target, L_target.transpose(1, 2))
            elif model_name == "TransitionKAN":
                # regime[8] + horizon[1] = 9
                regime = X_batch[:, :8]
                horizon = X_batch[:, 8:9]
                y_pred = model(regime, horizon)
            else:
                y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)
        pbar.set_postfix({"loss": f"{epoch_loss:.4f}"})

    return losses


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train KAN Knowledge Store")
    parser.add_argument("--config", default="configs/options_dsl.yaml", help="Config file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--checkpoint", default="reports/options/kan_store", help="Checkpoint dir")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--synthetic-only", action="store_true", help="Use synthetic data only")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create checkpoint dir
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training KAN Knowledge Store")
    print(f"  Config: {args.config}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print()

    # Create KAN config
    kan_config = KANStoreConfig()

    # Generate synthetic training data
    print("Generating synthetic training data...")
    vol_specs = config["training"]["vol_surface"]

    X_vol, y_vol = generate_synthetic_vol_surface_data(
        config["training"]["num_vol_surface_samples"],
        vol_specs,
        device=args.device
    )
    X_cov, y_cov = generate_synthetic_covariance_data(
        config["training"]["num_covariance_samples"],
        device=args.device
    )
    X_trans, y_trans = generate_synthetic_transition_data(
        config["training"]["num_transition_samples"],
        device=args.device
    )
    print(f"  Vol surface: X={X_vol.shape}, y={y_vol.shape}")
    print(f"  Covariance: X={X_cov.shape}, y={y_cov.shape}")
    print(f"  Transition: X={X_trans.shape}, y={y_trans.shape}")
    print()

    # Create dataloaders
    vol_dataset = TensorDataset(X_vol, y_vol)
    cov_dataset = TensorDataset(X_cov, y_cov)
    trans_dataset = TensorDataset(X_trans, y_trans)

    vol_loader = DataLoader(vol_dataset, batch_size=args.batch_size, shuffle=True)
    cov_loader = DataLoader(cov_dataset, batch_size=args.batch_size, shuffle=True)
    trans_loader = DataLoader(trans_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize KAN Knowledge Store
    print("Initializing KAN Knowledge Store...")
    store = KANKnowledgeStore(config=kan_config, device=args.device)
    print("  VolSurfaceKAN initialized")
    print("  CovarianceKAN initialized")
    print("  TransitionKAN initialized")
    print()

    # Train models
    print("Training KAN models...")
    vol_losses = train_kan_model(
        store.vol_surface, vol_loader, args.device, args.epochs, args.learning_rate, "VolSurfaceKAN"
    )
    cov_losses = train_kan_model(
        store.covariance, cov_loader, args.device, args.epochs, args.learning_rate, "CovarianceKAN"
    )
    trans_losses = train_kan_model(
        store.transition, trans_loader, args.device, args.epochs, args.learning_rate, "TransitionKAN"
    )
    print()

    # Save store
    print("Saving checkpoint...")
    store.save(str(checkpoint_dir))

    # Save loss history
    losses_file = checkpoint_dir / "losses.json"
    losses_data = {
        "vol_surface": vol_losses,
        "covariance": cov_losses,
        "transition": trans_losses,
    }
    with open(losses_file, "w") as f:
        json.dump(losses_data, f, indent=2)

    print(f"Saved checkpoint to {checkpoint_dir}")
    print(f"Saved loss history to {losses_file}")
    print()

    # Print final losses
    print("Training complete!")
    print(f"  VolSurfaceKAN final loss: {vol_losses[-1]:.6f}")
    print(f"  CovarianceKAN final loss: {cov_losses[-1]:.6f}")
    print(f"  TransitionKAN final loss: {trans_losses[-1]:.6f}")


if __name__ == "__main__":
    main()
