"""
Manifold Mapper: Deep integration between natural language narratives and geometric manifolds.

Maps news text to suggested Geometric Algebra (GA) states and Ricci curvature features.
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

    def map_news_to_state(self, news_text: str) -> Dict[str, Any]:
        """
        Analyze news text and suggest a geometric state for the market manifold.

        Args:
            news_text: Market news or narrative

        Returns:
            Dict containing:
                - regime: Suggested regime name (STABLE, STRESS, etc.)
                - ga_rotor_magnitude: Suggested rotor intensity
                - ricci_curvature: Suggested manifold curvature
                - reasoning: Brief LLM reasoning for the mapping
        """
        system_prompt = self._build_mapping_system_prompt()
        user_message = f"NEWS ARTICLE:\n{news_text}\n\nSUGGEST MANIFOLD STATE:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = self.client.chat(
                messages=messages,
                temperature=0.3,  # Lower for more analytical mapping
                max_tokens=500,
            )

            # Parse JSON from response
            # LLM should output pure JSON per instructions
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
        """Build system prompt for News-to-Manifold mapping."""
        return """You are a Quantitative Geometric Analyst. Your job is to translate market news narratives into the language of differential geometry and Clifford Algebra.

You will output a JSON object representing the 'Intrinsic State' of the market manifold.

MAPPING RULES:
1. 'STRESS' Regime: Triggered by shocks, geopolitical conflict, inflation spikes, supply chain breakage.
   - ricci_mean: -25.0 to -50.0 (Deep negative curvature)
   - ga_rotor_magnitude: 0.8 to 1.5 (High rotation)
2. 'TRANSITION' Regime: Triggered by trend reversals, diverging signals, or 'uncertainty'.
   - ricci_mean: -10.0 to -25.0
   - ga_rotor_magnitude: 0.4 to 0.8
3. 'STABLE' Regime: Growth, low inflation, predictability.
   - ricci_mean: -5.0 to 5.0 (Flat/Positive)
   - ga_rotor_magnitude: 0.05 to 0.3
4. 'RECOVERY' Regime: Stabilization after crash, absorption of shocks.
   - ricci_mean: -10.0 to 0.0
   - ga_rotor_magnitude: 0.2 to 0.5

OUTPUT FORMAT (JSON):
{
  "regime": "STRESS" | "TRANSITION" | "STABLE" | "RECOVERY",
  "ga_rotor_magnitude": float,
  "ga_bivector_energy": float (typically 10-20x rotor),
  "ricci_mean_core_60d": float,
  "mst_stress_core_60d": float (0.1 to 0.9),
  "realized_vol_20d": float (0.1 to 0.8),
  "reasoning": "string"
}

Be analytical and precise."""

if __name__ == "__main__":
    # Quick test
    mapper = ManifoldMapper()
    test_news = "Silver prices soar as COMEX physical inventory vanishes; massive supply deficit expected for 2026."
    try:
        state = mapper.map_news_to_state(test_news)
        print(json.dumps(state, indent=2))
    except Exception as e:
        print(f"Error: {e}")
