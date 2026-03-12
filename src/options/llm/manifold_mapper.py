"""
Manifold Mapper: Deep integration between natural language narratives, 
prior institutional context, and geometric manifolds.
"""

import json
from typing import Dict, Any, Optional, List
from .client import OllamaClient


class ManifoldMapper:
    """Maps text narratives to geometric manifold parameters."""

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()

    def map_news_to_state(self, news_text: str, prior_context: Optional[str] = None) -> Dict[str, Any]:
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
            response = self.client.chat(messages=messages, temperature=0.3, max_tokens=600)
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > 0:
                return json.loads(response[start_idx:end_idx])
            else:
                raise ValueError(f"Could not find JSON in LLM response.")
        except Exception as e:
            raise RuntimeError(f"Manifold mapping failed: {e}")

    def _build_mapping_system_prompt(self) -> str:
        return """You are a Quantitative Geometric Analyst. Translate market news narratives into the language of differential geometry and Clifford Algebra.

MAPPING RULES:
1. 'STRESS' Regime: ricci_mean: -25.0 to -50.0, ga_rotor_magnitude: 0.8 to 1.5
2. 'TRANSITION' Regime: ricci_mean: -10.0 to -25.0, ga_rotor_magnitude: 0.4 to 0.8
3. 'STABLE' Regime: ricci_mean: -5.0 to 5.0, ga_rotor_magnitude: 0.05 to 0.3
4. 'RECOVERY' Regime: ricci_mean: -10.0 to 0.0, ga_rotor_magnitude: 0.2 to 0.5

OUTPUT FORMAT (JSON):
{
  "regime": "STRESS" | "TRANSITION" | "STABLE" | "RECOVERY",
  "ga_rotor_magnitude": float,
  "ricci_mean_core_60d": float,
  "mst_stress_core_60d": float,
  "realized_vol_20d": float,
  "reasoning": "string"
}"""


class TemporalManifoldMapper:
    """Maps multi-scale temporal narratives to geometric manifold parameters."""

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()

    def map_to_state(self, current_news: str, priors: Dict[str, str], asset: str = "silver") -> Dict[str, Any]:
        system_prompt = self._build_temporal_system_prompt()
        context_str = "\n".join([f"[{scale} PRIOR]: {text}" for scale, text in priors.items()])
        user_message = f"TEMPORAL CONTEXT:\n{context_str}\n\nLATEST NEWS (T-0):\n{current_news}\n\nRECONCILE AND SUGGEST STATE:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = self.client.chat(messages=messages, temperature=0.2, max_tokens=800)
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > 0:
                return json.loads(response[start_idx:end_idx])
            else:
                raise ValueError(f"No JSON found in LLM response.")
        except Exception as e:
            raise RuntimeError(f"Temporal manifold mapping failed: {e}")

    def _build_temporal_system_prompt(self) -> str:
        return """You are a Senior Quantitative Strategist. Reconcile multi-scale temporal context to suggest a manifold state.

HIERARCHICAL RECONCILIATION RULES:
1. T-360 (Annual): Sets the 'Manifold Anchor' (Ricci Curvature).
2. T-7 (Weekly): Sets the 'Momentum Vector' (Rotor starting state).
3. Model Views: Calibrates Greeks.
4. T-0 (News): Triggers the 'Instantaneous Shift' (Rotor Spike).

OUTPUT FORMAT (JSON):
{
  "regime": "STABLE" | "TRANSITION" | "STRESS" | "RECOVERY",
  "ga_rotor_magnitude": float,
  "ricci_mean_core_60d": float,
  "realized_vol_20d": float,
  "contextual_calibration": "Explanation of how T-360 anchored the view while T-0 shocked it.",
  "greek_adjustment": "Directional bias for Delta/Vega."
}"""
