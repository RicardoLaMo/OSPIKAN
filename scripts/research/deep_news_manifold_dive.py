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

def deep_news_manifold_dive(news_text: str):
    print("="*80)
    print("DEEP DIVE: NEWS NARRATIVE TO GEOMETRIC MANIFOLD")
    print("="*80)
    
    # 1. INITIALIZE LLM MAPPER
    print("\n[STEP 1] LLM ANALYZING NARRATIVE...")
    # Using the live Ollama client we verified earlier
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5-coder:latest")
    mapper = ManifoldMapper(client=client)
    
    try:
        # LLM maps text to specific geometric coordinates
        manifold_state = mapper.map_news_to_state(news_text)
        print(f"   ✓ NARRATIVE MAPPED TO MANIFOLD STATE:")
        print(json.dumps(manifold_state, indent=4))
    except Exception as e:
        print(f"   ✗ Mapping failed: {e}")
        return

    # 2. INITIALIZE HIGH-FIDELITY KAN
    print("\n[STEP 2] PROJECTING NARRATIVE STATE TO VOLSURFACEKAN...")
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

    # 3. PREPARE FEATURE VECTOR FOR KAN
    # [ricci_mean, ricci_min, mst, rotor, vol, mom, p0, p1]
    # We map the LLM's suggested coordinates into the bridge format
    regime_feat_dict = {
        'ricci_mean_core_60d': manifold_state.get('ricci_mean_core_60d', -19.0),
        'ricci_min_core_60d': manifold_state.get('ricci_mean_core_60d', -19.0) * 2.0, # Proxy
        'mst_stress_core_60d': manifold_state.get('mst_stress_core_60d', 0.5),
        'ga_rotor_magnitude_60d': manifold_state.get('ga_rotor_magnitude', 0.5),
        'realized_vol_20d': manifold_state.get('realized_vol_20d', 0.2),
        'momentum_10d': 0.0,
        'p_regime_0': 0.1 if manifold_state['regime'] == 'STRESS' else 0.8,
        'p_regime_1': 0.8 if manifold_state['regime'] == 'STRESS' else 0.1,
    }

    # 4. INFERENCE
    strikes = [0.9, 1.0, 1.1]
    results = []
    
    with torch.no_grad():
        for K_ratio in strikes:
            res = pricer.price(
                spot=100.0,
                strike=100.0 * K_ratio,
                time_to_expiry=30/365,
                volatility='kan_regime',
                regime_features=regime_feat_dict
            )
            results.append(res.used_volatility)

    # 5. FINAL REPORT
    print("\n[STEP 3] NARRATIVE-IMPLIED VOLATILITY SMILE:")
    print(f"   News Impact: {manifold_state['reasoning']}")
    print("-" * 50)
    print(f"   - 90% Put Vol:  {results[0]*100:.2f}%")
    print(f"   - 100% ATM Vol: {results[1]*100:.2f}%")
    print(f"   - 110% Call Vol: {results[2]*100:.2f}%")
    print("-" * 50)
    
    skew = (results[0] - results[2]) * 100
    print(f"   Calculated Narrative Skew: {skew:.2f}%")
    print("\n" + "="*80)
    print("SUCCESS: Text Narrative fully projected onto High-Fidelity Manifold.")

if __name__ == "__main__":
    # Test case: Geopolitical Tension in the Middle East causing Silver supply fears
    news = """
    URGENT: Major escalation in the Middle East as regional powers exchange missile fire. 
    Global shipping lanes in the Strait of Hormuz are effectively blocked. 
    Silver market analysts warn of 'unprecedented supply squeeze' as physical transport stops.
    Silver spot prices are up 8% in pre-market trading.
    """
    deep_news_manifold_dive(news)
