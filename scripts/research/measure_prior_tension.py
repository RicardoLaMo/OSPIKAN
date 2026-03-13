import sys
import os
import torch
import numpy as np
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.options.kan_store.store import KANKnowledgeStore
from src.options.pricing.regime_adjusted import RegimeAdjustedPricer
from src.options.llm.manifold_mapper import TemporalManifoldMapper
from src.options.llm.client import OllamaClient
from src.geometry.optimal_transport import ManifoldTensionValidator

def measure_refined_tension():
    print("="*85)
    print("DUAL MANIFOLD TENSION: LEVEL (W2) vs STRUCTURE (GW)")
    print("="*85)

    # 1. SETUP DATA STATE
    data_state = {
        'ricci_mean_core_60d': -2.0,
        'ga_rotor_magnitude_60d': 0.1,
        'realized_vol_20d': 0.18,
        'p_regime_0': 0.8, 'p_regime_1': 0.1
    }

    # 2. SETUP REPORT STATE (SHOCKED)
    annual_report = "Structural physical deficit in Silver; expecting multi-year supply squeeze."
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5-coder:latest")
    mapper = TemporalManifoldMapper(client=client)
    
    try:
        report_state_raw = mapper.map_to_state(current_news="Supply shock scenario.", priors={"T-360": annual_report})
        report_state = {
            'ricci_mean_core_60d': report_state_raw.get('ricci_mean_core_60d', -30.0),
            'ga_rotor_magnitude_60d': report_state_raw.get('ga_rotor_magnitude', 1.2),
            'realized_vol_20d': 0.45,
            'p_regime_0': 0.1, 'p_regime_1': 0.8
        }
    except Exception:
        return

    # 3. PROJECT TO MANIFOLD
    store = KANKnowledgeStore()
    checkpoint_path = PROJECT_ROOT / "reports/options/kan_store/"
    if checkpoint_path.exists():
        store.load(str(checkpoint_path))
        store.feature_bridge.mean = torch.tensor([[-19.0, -40.0, 0.5, 0.5, 0.2, 0.0, 0.5, 0.5]])
        store.feature_bridge.std = torch.tensor([[5.0, 10.0, 0.2, 0.5, 0.2, 0.05, 0.3, 0.3]])
    
    pricer = RegimeAdjustedPricer(kan_store=store)
    strikes = np.linspace(0.8, 1.2, 20)
    cloud_data, cloud_report = [], []
    
    with torch.no_grad():
        for s in strikes:
            res_d = pricer.price(100.0, 100.0*s, 30/365, 'kan_regime', regime_features=data_state)
            cloud_data.append([s, res_d.used_volatility])
            res_r = pricer.price(100.0, 100.0*s, 30/365, 'kan_regime', regime_features=report_state)
            cloud_report.append([s, res_r.used_volatility])

    # 4. COMPUTE DUAL METRICS
    validator = ManifoldTensionValidator()
    metrics = validator.compute_full_tension_report(np.array(cloud_data), np.array(cloud_report))

    # 5. FINAL REPORT
    print("\n" + "="*85)
    print("REFINED MANIFOLD TENSION REPORT")
    print("-" * 85)
    print(f"1. LEVEL TENSION (Standard W2):    {metrics['level_tension_w2']:.6f}")
    print("   Interpretation: Absolute difference in market 'fear' (volatility levels).")
    print(f"\n2. STRUCTURAL TENSION (Gromov-W): {metrics['structural_tension_gw']:.6f}")
    print("   Interpretation: Geometric deformation of the market 'shape' (isometry).")
    print("-" * 85)
    
    if metrics['level_tension_w2'] > 0.05 and metrics['structural_tension_gw'] < 0.005:
        print("DIAGNOSIS: PURE VOLATILITY EXPANSION.")
        print("The market is becoming more expensive, but the fundamental correlation")
        print("structure remains intact (Parallel Shift).")
    elif metrics['structural_tension_gw'] > 0.01:
        print("DIAGNOSIS: STRUCTURAL REGIME BREAK.")
        print("The narrative implies a fundamental change in market geometry.")
        print("Expect sharp skew inversions or non-linear correlation snaps.")
    print("="*85)

if __name__ == "__main__":
    measure_refined_tension()
