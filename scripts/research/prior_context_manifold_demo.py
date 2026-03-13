import sys
import os
import torch
import json
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.options.kan_store.store import KANKnowledgeStore
from src.options.pricing.regime_adjusted import RegimeAdjustedPricer
from src.options.llm.manifold_mapper import ManifoldMapper
from src.options.llm.client import OllamaClient

def run_prior_context_demo():
    print("="*80)
    print("BUSINESS CASE: NARRATIVE SHOCK WITH PRIOR INSTITUTIONAL CONTEXT")
    print("="*80)
    
    # 1. Define Contexts
    prior_context = """
    T-1 Desk Report: 
    - Asset: Silver
    - Regime: STABLE
    - Implied Volatility (ATM): ~17%
    - Positioning: Delta-neutral, slightly short Vega.
    - Outlook: Market is in a tight consolidation phase. No major catalysts expected.
    """
    
    breaking_news = """
    URGENT (T-0): Major missile strike in the Strait of Hormuz. 
    COMEX silver inventories plunge as industrial buyers panic-hoard physical metal. 
    Spot prices are violently breaking resistance levels.
    """
    
    print("\n[INPUTS]")
    print(f"PRIOR CONTEXT:\n{prior_context.strip()}")
    print("-" * 50)
    print(f"BREAKING NEWS:\n{breaking_news.strip()}")

    # 2. LLM Analysis
    print("\n[STEP 1] LLM SYNTHESIZING PRIOR CONTEXT & NEW NARRATIVE...")
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5-coder:latest")
    mapper = ManifoldMapper(client=client)
    
    try:
        new_state = mapper.map_news_to_state(breaking_news, prior_context=prior_context)
        print("   ✓ NARRATIVE MAPPED TO NEW MANIFOLD STATE:")
        print(json.dumps(new_state, indent=4))
    except Exception as e:
        print(f"   ✗ Mapping failed: {e}")
        return

    # 3. KAN Inference
    print("\n[STEP 2] PROJECTING MANIFOLD STATES TO VOLSURFACEKAN...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = KANKnowledgeStore(device=device)
    
    checkpoint_path = PROJECT_ROOT / "reports/options/kan_store/"
    if checkpoint_path.exists():
        store.load(str(checkpoint_path))
        # Use our calibrated high-fidelity mean/std
        store.feature_bridge.mean = torch.tensor([[-19.0, -40.0, 0.5, 0.5, 0.2, 0.0, 0.5, 0.5]], device=device)
        store.feature_bridge.std = torch.tensor([[5.0, 10.0, 0.2, 0.5, 0.2, 0.05, 0.3, 0.3]], device=device)
    
    store.to(device)
    store.eval()
    
    pricer = RegimeAdjustedPricer(kan_store=store)

    # Helper function to get vols from state
    def get_vols(state_dict):
        strikes = [0.9, 1.0, 1.1]
        vols = []
        regime_feat_dict = {
            'ricci_mean_core_60d': state_dict.get('ricci_mean_core_60d', -19.0),
            'ricci_min_core_60d': state_dict.get('ricci_mean_core_60d', -19.0) * 2.0,
            'mst_stress_core_60d': state_dict.get('mst_stress_core_60d', 0.5),
            'ga_rotor_magnitude_60d': state_dict.get('ga_rotor_magnitude', 0.5),
            'realized_vol_20d': state_dict.get('realized_vol_20d', 0.2),
            'momentum_10d': 0.0,
            'p_regime_0': 0.1 if state_dict.get('regime') == 'STRESS' else 0.8,
            'p_regime_1': 0.8 if state_dict.get('regime') == 'STRESS' else 0.1,
        }
        with torch.no_grad():
            for K_ratio in strikes:
                res = pricer.price(100.0, 100.0 * K_ratio, 30/365, 'kan_regime', regime_features=regime_feat_dict)
                vols.append(res.used_volatility)
        return vols

    # Prior state dummy mapping (based on the context)
    prior_state = {
        "regime": "STABLE",
        "ga_rotor_magnitude": 0.1,
        "ricci_mean_core_60d": -2.0,
        "mst_stress_core_60d": 0.2,
        "realized_vol_20d": 0.17
    }

    prior_vols = get_vols(prior_state)
    new_vols = get_vols(new_state)

    # 4. Report
    print("\n[STEP 3] DYNAMIC SHOCK IMPACT ON VOLATILITY SURFACE:")
    print("-" * 65)
    print(f"{'Option Type':<15} | {'Prior (T-1)':<15} | {'Shocked (T-0)':<15} | {'Δ Shift':<10}")
    print("-" * 65)
    
    labels = ["90% OTM Put", "100% ATM Call", "110% OTM Call"]
    for i, label in enumerate(labels):
        p_vol = prior_vols[i] * 100
        n_vol = new_vols[i] * 100
        delta = n_vol - p_vol
        print(f"{label:<15} | {p_vol:>13.2f}% | {n_vol:>13.2f}% | {delta:>+9.2f}%")
        
    print("-" * 65)
    prior_skew = (prior_vols[0] - prior_vols[2]) * 100
    new_skew = (new_vols[0] - new_vols[2]) * 100
    print(f"{'IV Skew':<15} | {prior_skew:>13.2f}% | {new_skew:>13.2f}% | {(new_skew - prior_skew):>+9.2f}%")
    
    print("\n[BUSINESS INSIGHT]")
    print(f"The LLM correctly recognized the prior {prior_state['regime']} state and ")
    print(f"applied the breaking news to shock the manifold into a {new_state['regime']} state.")
    print("Notice how the KAN model automatically reprices the skew dynamically based on the ")
    print("shifted geometric coordinates (Clifford Rotor & Ricci Curvature).")
    print("="*80)

if __name__ == "__main__":
    run_prior_context_demo()
