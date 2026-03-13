import json
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.options.llm.client import OllamaClient
from src.options.llm.manifold_mapper import TemporalManifoldMapper


def run_hierarchical_prior_demo():
    print("=" * 80)
    print("HIERARCHICAL TEMPORAL PRIOR RECONCILIATION")
    print("=" * 80)

    priors = {
        "T-360 (Annual)": "Strategic Outlook: Silver entering 6th year of structural deficit. Target $120. Manifold anchor is deep negative curvature due to physical supply inelasticity.",
        "T-7 (Weekly)": "Tactical Desk View: Momentum is slowing as prices hit $85 resistance. Gamma-hedging flows are stabilizing the IV at 25%. Expecting local mean-reversion.",
        "Model Conclusion (Quant)": "GARCH(1,1) suggests vol decay. Black-Litterman view is neutral on direction but overweight on tail-risk (kurtosis).",
    }

    breaking_news = "T-0: Major inventory withdrawal at COMEX; 15% of physical stock moved to private vaults overnight. Rotor sensor spiking."

    print("\n[TEMPORAL PRIORS INGESTED]")
    for scale, text in priors.items():
        print(f"-> {scale}: {text}")
    print(f"\n[NEW T-0 INPUT]: {breaking_news}")

    print("\n[STEP 1] LLM SYNTHESIZING SCALES (Strategic Anchor vs. Tactical Shock)...")
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5-coder:latest")
    mapper = TemporalManifoldMapper(client=client)

    try:
        calibrated_state = mapper.map_to_state(breaking_news, priors=priors)
        print("\n   ✓ CALIBRATED MANIFOLD STATE (RECONCILED):")
        print(json.dumps(calibrated_state, indent=4))
    except Exception as e:
        print(f"   ✗ Synthesis failed: {e}")
        return

    print("\n" + "=" * 80)
    print("BUSINESS CASE VALIDATION:")
    print("-" * 80)
    print(f"Strategic Anchor (Ricci): {calibrated_state['ricci_mean_core_60d']}")
    print(f"Tactical Shock (Rotor):   {calibrated_state['ga_rotor_magnitude']}")
    print("-" * 80)
    print(f"TEMPORAL LOGIC: {calibrated_state['contextual_calibration']}")
    print(f"GREEK PERSPECTIVE: {calibrated_state['greek_adjustment']}")
    print("=" * 80)


if __name__ == "__main__":
    run_hierarchical_prior_demo()
