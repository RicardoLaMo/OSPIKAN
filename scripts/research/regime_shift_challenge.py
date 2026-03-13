import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.optimal_transport import GromovWassersteinValidator
from src.options.kan_store.store import KANKnowledgeStore
from src.physics.market_pdes import BurgersEquation, PDEConfig


class BaselineMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_layers=[32, 32, 16], output_dim=1):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, regime, log_moneyness, time_to_expiry):
        x = torch.cat([regime, log_moneyness, time_to_expiry], dim=1)
        output = self.net(x)
        return 2.0 * torch.sigmoid(output)


def train_baseline(kan_store: KANKnowledgeStore, epochs=100):
    print("Training Baseline MLP to fit normal market regime...")
    model = BaselineMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    n_samples = 2000
    regimes = torch.zeros(n_samples, 8)
    regimes[:, 0] = torch.normal(-19.0, 2.0, (n_samples,))
    regimes[:, 1] = regimes[:, 0] * 1.5
    regimes[:, 2] = torch.normal(0.5, 0.1, (n_samples,))
    regimes[:, 3] = torch.normal(0.2, 0.05, (n_samples,))
    regimes[:, 4] = torch.normal(0.2, 0.05, (n_samples,))
    regimes[:, 5] = torch.normal(0.0, 0.01, (n_samples,))
    regimes[:, 6] = 0.8
    regimes[:, 7] = 0.1

    log_moneyness = torch.empty(n_samples, 1).uniform_(-0.2, 0.2)
    time_to_expiry = torch.empty(n_samples, 1).uniform_(0.01, 1.0)

    regimes_norm = kan_store.feature_bridge.transform(regimes)

    with torch.no_grad():
        targets = kan_store.vol_surface(regimes_norm, log_moneyness, time_to_expiry)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(regimes_norm, log_moneyness, time_to_expiry)
        loss = nn.MSELoss()(preds, targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:3d} | Loss: {loss.item():.6f}")

    model.eval()
    return model


def compute_pde_residual(model, regime_norm, strikes, t_exp, pde: BurgersEquation):
    log_moneyness = torch.tensor(np.log(strikes), dtype=torch.float32).unsqueeze(-1).requires_grad_(True)
    time_tensor = torch.tensor([t_exp] * len(strikes), dtype=torch.float32).unsqueeze(-1).requires_grad_(True)
    regime_expanded = regime_norm.expand(len(strikes), -1).requires_grad_(True)

    u = model(regime_expanded, log_moneyness, time_tensor)
    u_t = torch.autograd.grad(u, time_tensor, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, log_moneyness, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, log_moneyness, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    residual = pde.residual(u, u_t, u_x, u_xx)
    return torch.mean(residual**2).item()


def measure_latency(model, regime_norm, n_points=100):
    log_moneyness = torch.zeros(n_points, 1)
    time_tensor = torch.ones(n_points, 1) * 0.1
    regime_expanded = regime_norm.expand(n_points, -1)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            _ = model(regime_expanded, log_moneyness, time_tensor)
    end = time.perf_counter()
    return (end - start) / 10 * 1000


def run_challenge():
    print("=" * 80)
    print("REGIME-SHIFT FIDELITY CHALLENGE: Baseline MLP vs SPIKAN-GA")
    print("=" * 80)

    store = KANKnowledgeStore()
    checkpoint_path = PROJECT_ROOT / "reports/options/kan_store/"
    if checkpoint_path.exists():
        store.load(str(checkpoint_path))
        store.feature_bridge.mean = torch.tensor([[-19.0, -30.0, 0.5, 0.2, 0.2, 0.0, 0.8, 0.1]])
        store.feature_bridge.std = torch.tensor([[5.0, 10.0, 0.2, 0.1, 0.1, 0.05, 0.1, 0.05]])
    else:
        print("Error: KAN store not found.")
        return

    baseline = train_baseline(store)

    shock_state = {
        'ricci_mean_core_60d': -45.0,
        'ricci_min_core_60d': -70.0,
        'mst_stress_core_60d': 0.9,
        'ga_rotor_magnitude_60d': 1.5,
        'realized_vol_20d': 0.75,
        'momentum_10d': 0.05,
        'p_regime_0': 0.05,
        'p_regime_1': 0.9
    }

    regime_raw = torch.tensor([[shock_state[col] for col in store.feature_bridge.feature_cols]])
    regime_norm = store.feature_bridge.transform(regime_raw)

    hist_strikes = np.linspace(0.8, 1.2, 25)
    historical_cloud = []
    for s in hist_strikes:
        v = 0.7 + 1.5 * (max(0, 1.0 - s)**2.0) + 0.2 * abs(1.0 - s)
        historical_cloud.append([s, v])
    historical_cloud = np.array(historical_cloud)

    models = {
        "Baseline MLP": baseline,
        "SPIKAN-GA (Challenger)": store.vol_surface
    }

    results = []
    strikes = np.linspace(0.8, 1.2, 20)
    validator = GromovWassersteinValidator(epsilon=0.01)
    pde = BurgersEquation(PDEConfig(viscosity=0.05))

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        cloud = []
        with torch.no_grad():
            for s in strikes:
                log_m = torch.tensor([[np.log(s)]], dtype=torch.float32)
                t_exp = torch.tensor([[30 / 365]], dtype=torch.float32)
                v = model(regime_norm, log_m, t_exp).item()
                cloud.append([s, v])
        cloud = np.array(cloud)

        _, gw_dist = validator.validate_isometry(cloud, historical_cloud)
        pde_res = compute_pde_residual(model, regime_norm, strikes, 30 / 365, pde)
        latency = measure_latency(model, regime_norm)

        results.append({
            "Model": name,
            "GW-OT Dist (2011 Peak)": gw_dist,
            "PDE Residual (Burgers)": pde_res,
            "Latency (ms)": latency
        })

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("FINAL COMPARISON TABLE")
    print("-" * 80)
    print(df_res.to_string(index=False))
    print("=" * 80)

    df_res.to_csv("regime_shift_challenge_results.csv", index=False)
    print("\nResults saved to regime_shift_challenge_results.csv")


if __name__ == "__main__":
    run_challenge()
