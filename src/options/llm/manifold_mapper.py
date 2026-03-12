"""
Manifold Mapper: Deep integration between natural language narratives, prior institutional context, and geometric manifolds.

Maps news text and prior analytics reports to suggested Geometric Algebra (GA) states and Ricci curvature features.
"""

import json
from typing import Dict, Any, Optional
from .client import OllamaClient


class ManifoldMapper:
    """Maps text narratives to geometric manifold parameters."""

    def __init__(self, client: Optional[OllamaClient] = None):
        """
        Initialize mapper.

        Args:
            client: OllamaClient (default: creates new one)
        """
        self.client = client or OllamaClient()

    def map_news_to_state(self, news_text: str, prior_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze news text against prior context and suggest a geometric state for the market manifold.

        Args:
            news_text: Market news or narrative
            prior_context: Optional text containing prior institutional analytics, Greeks, or desk views.

        Returns:
            Dict containing:
                - regime: Suggested regime name (STABLE, STRESS, etc.)
                - ga_rotor_magnitude: Suggested rotor intensity
                - ricci_curvature: Suggested manifold curvature
                - reasoning: Brief LLM reasoning for the mapping, contrasting with prior context
        """
        system_prompt = self._build_mapping_system_prompt()
        
        user_message = ""
        if prior_context:
            user_message += f"PRIOR INSTITUTIONAL CONTEXT:\n{prior_context}\n\n"
            
        user_message += f"NEW BREAKING NARRATIVE:\n{news_text}\n\nSUGGEST NEW MANIFOLD STATE:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = self.client.chat(
                messages=messages,
                temperature=0.3,  # Lower for more analytical mapping
                max_tokens=600,
            )

            # Parse JSON from response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError(f"Could not find JSON in LLM response: {response}")

        except Exception as e:
            raise RuntimeError(f"Manifold mapping failed: {e}")

    def _build_mapping_system_prompt(self) -> str:
        """Build system prompt for News-to-Manifold mapping with prior context."""
        return """You are a Quantitative Geometric Analyst at a top-tier investment bank. Your job is to translate breaking market news narratives into the language of differential geometry and Clifford Algebra, taking into account any Prior Institutional Context (previous Greeks, IV levels, desk positioning).

You will output a JSON object representing the 'Intrinsic State' of the market manifold.

MAPPING RULES:
1. 'STRESS' Regime: Triggered by shocks, geopolitical conflict, inflation spikes.
   - ricci_mean_core_60d: -25.0 to -50.0 (Deep negative curvature)
   - ga_rotor_magnitude: 0.8 to 1.5 (High rotation)
2. 'TRANSITION' Regime: Triggered by trend reversals, diverging signals, or 'uncertainty'.
   - ricci_mean_core_60d: -10.0 to -25.0
   - ga_rotor_magnitude: 0.4 to 0.8
3. 'STABLE' Regime: Growth, predictability, or continuing the prior calm state.
   - ricci_mean_core_60d: -5.0 to 5.0 (Flat/Positive)
   - ga_rotor_magnitude: 0.05 to 0.3
4. 'RECOVERY' Regime: Stabilization after crash, absorption of shocks.
   - ricci_mean_core_60d: -10.0 to 0.0
   - ga_rotor_magnitude: 0.2 to 0.5

If Prior Context is provided, calculate the 'shock' severity relative to that baseline. For instance, if prior IV was 15% (STABLE) and news indicates a massive supply squeeze, the rotor magnitude should spike severely (e.g., > 1.0).

OUTPUT FORMAT (JSON):
{
  "regime": "STRESS" | "TRANSITION" | "STABLE" | "RECOVERY",
  "ga_rotor_magnitude": float,
  "ga_bivector_energy": float,
  "ricci_mean_core_60d": float,
  "mst_stress_core_60d": float (0.1 to 0.9),
  "realized_vol_20d": float (0.1 to 0.8),
  "reasoning": "Explain how the breaking news shifts the manifold relative to the prior context."
}

Be analytical and precise."""

if __name__ == "__main__":
    # Quick test
    mapper = ManifoldMapper()
    test_prior = "Desk Report (T-1): Silver market in tight consolidation. IV at 17%, Delta flat. Expecting range-bound behavior."
    test_news = "Silver prices soar as COMEX physical inventory vanishes; massive supply deficit expected for 2026."
    try:
        state = mapper.map_news_to_state(test_news, test_prior)
        print(json.dumps(state, indent=2))
    except Exception as e:
        print(f"Error: {e}")
