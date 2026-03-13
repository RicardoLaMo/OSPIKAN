import sys
import os
import torch
import numpy as np
import json
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.options.kan_store.store import KANKnowledgeStore
from src.options.pricing.regime_adjusted import RegimeAdjustedPricer
from src.options.llm.manifold_mapper import TemporalManifoldMapper
from src.options.llm.client import OllamaClient
from src.geometry.optimal_transport import GromovWassersteinValidator

def measure_manifold_tension():
    print("="*85)
    print("BUSINESS CASE: MEASURING NARRATIVE TENSION VIA GROMOV-WASSERSTEIN TRANSPORT")
    print("="*85)

    # 1. THE DATA MANIFOLD (M_data)
    # What the quantitative sensors and current market prices are showing right now.
    data_state = {
        'ricci_mean_core_60d': -2.0,    # Flat
        'ricci_min_core_60d': -5.0,
        'mst_stress_core_60d': 0.3,     # Low stress
        'ga_rotor_magnitude_60d': 0.1,  # No rotation
        'realized_vol_20d': 0.18,       # Low vol
        'momentum_10d': 0.0,
        'p_regime_0': 0.8, 'p_regime_1': 0.1
    }
    print("\n[MANIFOLD A] DATA-DRIVEN STATE (Current Market Prices):")
    print("   -> State: STABLE (Rotor: 0.1, Vol: 18%)")

    # 2. THE REPORT MANIFOLD (M_report)
    # What the LLM extracts from the Annual Report / Prior Context
    annual_report = """
    T-360 Annual Outlook: The silver market is entering a severe structural deficit. 
    Above-ground physical inventories are depleted. We expect violent upside tail-risk 
    as industrial demand overwhelms paper market liquidity.
    """
    
    print("\n[MANIFOLD B] REPORT-DRIVEN STATE (Institutional Prior):")
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5-coder:latest")
    mapper = TemporalManifoldMapper(client=client)
    
    try:
        # We ask the LLM to map ONLY the report to a geometric state
        report_state_raw = mapper.map_to_state(
            current_news="Evaluate pure report baseline.", 
            priors={"T-360": annual_report}
        )
        print("   -> LLM Deciphered State:")
        print(f"      Regime: {report_state_raw.get('regime')}")
        print(f"      Rotor:  {report_state_raw.get('ga_rotor_magnitude')}")
        print(f"      Ricci:  {report_state_raw.get('ricci_mean_core_60d')}")
        
    except Exception as e:
        print(f"   ✗ LLM mapping failed: {e}")
        return

    report_state = {
        'ricci_mean_core_60d': report_state_raw.get('ricci_mean_core_60d', -30.0),
        'ricci_min_core_60d': report_state_raw.get('ricci_mean_core_60d', -30.0) * 2.0,
        'mst_stress_core_60d': 0.8,
        'ga_rotor_magnitude_60d': report_state_raw.get('ga_rotor_magnitude', 1.0),
        'realized_vol_20d': report_state_raw.get('realized_vol_20d', 0.4),
        'momentum_10d': 0.0,
        'p_regime_0': 0.1, 'p_regime_1': 0.8
    }

    # 3. GENERATE THE SURFACES VIA SPIKAN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = KANKnowledgeStore(device=device)
    checkpoint_path = PROJECT_ROOT / "reports/options/kan_store/"
    if checkpoint_path.exists():
        store.load(str(checkpoint_path))
        store.feature_bridge.mean = torch.tensor([[-19.0, -40.0, 0.5, 0.5, 0.2, 0.0, 0.5, 0.5]], device=device)
        store.feature_bridge.std = torch.tensor([[5.0, 10.0, 0.2, 0.5, 0.2, 0.05, 0.3, 0.3]], device=device)
    store.to(device)
    store.eval()
    
    pricer = RegimeAdjustedPricer(kan_store=store)
    
    strikes = np.linspace(0.8, 1.2, 20)
    cloud_data = []
    cloud_report = []
    
    with torch.no_grad():
        for s in strikes:
            # Data Surface
            res_d = pricer.price(100.0, 100.0*s, 30/365, 'kan_regime', regime_features=data_state)
            cloud_data.append([s, res_d.used_volatility])
            # Report Surface
            res_r = pricer.price(100.0, 100.0*s, 30/365, 'kan_regime', regime_features=report_state)
            cloud_report.append([s, res_r.used_volatility])
            
    cloud_data = np.array(cloud_data)
    cloud_report = np.array(cloud_report)

    # 4. MEASURE TENSION VIA GW-OT
    print("\n[STEP 3] COMPUTING GROMOV-WASSERSTEIN TENSION (M_data <--> M_report)...")
    validator = GromovWassersteinValidator(epsilon=0.01)
    gw_distance = validator.compute_relational_distance(cloud_data, cloud_report)
    
    # 5. BUSINESS INSIGHT (PDE POTENTIAL ENERGY)
    print("\n" + "="*85)
    print(f"MANIFOLD TENSION REPORT")
    print("-" * 85)
    print(f"Relational Distance (GW-OT): {gw_distance:.6f}")
    print("-" * 85)
    
    if gw_distance > 0.01:
        print("BUSINESS INSIGHT: HIGH TENSION DETECTED.")
        print("The geometric shape of the current market prices (M_data) diverges significantly")
        print("from the structural reality defined in the institutional prior (M_report).")
        print("\nPDE IMPLICATION:")
        print("Because the KAN is constrained by Market PDEs, this high Wasserstein distance")
        print("acts as 'Potential Energy'. As the PDE flow evolves, the Data Manifold will likely")
        print("experience a violent 'snap' to align with the Report Manifold. This represents a")
        print("massive Alpha opportunity to buy mispriced tail-risk before the convergence.")
    else:
        print("BUSINESS INSIGHT: LOW TENSION.")
        print("The market has already absorbed the institutional prior. The data manifold and")
        print("the report manifold are isometrically aligned.")
    print("="*85)

if __name__ == "__main__":
    measure_manifold_tension()
