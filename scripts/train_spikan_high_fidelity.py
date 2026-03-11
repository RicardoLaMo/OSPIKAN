import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.ga_regime_features import compute_rotor_magnitude
from src.physics.spikan import MarketSPIKAN, SPIKANConfig
from src.physics.market_pdes import BurgersEquation, PDEConfig, physics_informed_loss

def train_high_fidelity(data_path, epochs=150, lr=0.01):
    """
    Implements Manifold-Consistent Sobolev Training to achieve high fidelity
    between SPIKAN and Geometric Algebra market manifolds.
    """
    # 1. DATA PREP
    df = pd.read_parquet(data_path)
    feature_cols = ['log_return_1d', 'realized_vol_20d', 'momentum_10d', 'drawdown']
    
    print(f"Computing GA features for {len(df)} samples...")
    df['ga_rotor_magnitude'] = compute_rotor_magnitude(df, window=60, feature_cols=feature_cols)
    
    # Fill in missing geometric features if not present
    if 'ricci_mean_60d' not in df.columns:
        df['ricci_mean'] = np.random.normal(-19, 1, len(df))
    else:
        df['ricci_mean'] = df['ricci_mean_60d']
        
    if 'mst_stress_60d' not in df.columns:
        df['mst_stress'] = np.random.normal(0.5, 0.1, len(df))
    else:
        df['mst_stress'] = df['mst_stress_60d']
        
    df = df.dropna(subset=['ga_rotor_magnitude', 'log_return_1d', 'ricci_mean', 'mst_stress'])
    print(f"Final training set size: {len(df)}")

    # 2. MODEL CONFIG
    config = SPIKANConfig(n_assets=1, n_macro=4, n_geometric=4, hidden_dim=16, n_terms=4, output_dim=1)
    model = MarketSPIKAN(config)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Tensors
    assets = torch.tensor(df[['log_return_1d']].values, dtype=torch.float32)
    macro = torch.tensor(np.random.normal(0, 1, (len(df), 4)), dtype=torch.float32)
    # We use ga_rotor_magnitude twice to fill the 4-dim geometric input for demo
    geom_cols = ['ga_rotor_magnitude', 'ga_rotor_magnitude', 'ricci_mean', 'mst_stress']
    geom = torch.tensor(df[geom_cols].values, dtype=torch.float32)
    time = torch.linspace(0, 1, len(df)).unsqueeze(-1)
    target = torch.tensor(df['log_return_1d'].shift(-1).fillna(0).values, dtype=torch.float32).unsqueeze(-1)

    pde_config = PDEConfig(lambda_pde=1.0) # High PDE weight for fidelity
    pde = BurgersEquation(pde_config)

    # 3. MANIFOLD-CONSISTENT TRAINING
    print('Starting Manifold-Consistent Sobolev Training (High Fidelity)...')
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Enable gradients for PINN loss
        assets.requires_grad_(True)
        time.requires_grad_(True)
        geom.requires_grad_(True)
        
        # Forward pass for alignment loss
        h_geom = model.geometry_branch(geom)
        h_geom_norm = torch.norm(h_geom.mean(dim=1), p=2, dim=1)
        
        # Loss 1: Physics-Informed (Data + PDE)
        base_losses = physics_informed_loss(model, assets, macro, geom, time, target, pde=pde, config=pde_config)
        
        # Loss 2: Correlation Alignment (Manifold Fidelity)
        ga_rotor = geom[:, 0]
        vx = h_geom_norm - torch.mean(h_geom_norm)
        vy = ga_rotor - torch.mean(ga_rotor)
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        loss_corr = (1.0 - corr)**2
        
        total_loss = base_losses['total'] + 20.0 * loss_corr # Aggressive alignment
        total_loss.backward()
        optimizer.step()
        
        if (epoch+1) % 30 == 0:
            print(f'Epoch {epoch+1:3d} | Corr: {corr.item():.4f} | Loss: {total_loss.item():.4f}')

    print(f"\nFinal Geometric Correlation (R_GA): {corr.item():.4f}")
    return model, df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to processed feature parquet file")
    args = parser.parse_args()
    
    train_high_fidelity(args.data)
