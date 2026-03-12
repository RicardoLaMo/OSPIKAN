"""
Temporal Manifold Mapper: Hierarchical synthesis of Annual, Monthly, and Weekly reports
to calibrate the SPIKAN-GA manifold.
"""

import json
from typing import Dict, Any, Optional, List
from .client import OllamaClient


class TemporalManifoldMapper:
    """Maps multi-scale temporal narratives to geometric manifold parameters."""

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()

    def map_to_state(
        self, 
        current_news: str, 
        priors: Dict[str, str], # {"T-360": "...", "T-7": "...", "model_view": "..."}
        asset: str = "silver"
    ) -> Dict[str, Any]:
        """
        Synthesize hierarchical context into a unified manifold state.
        """
        system_prompt = self._build_temporal_system_prompt()
        
        context_str = "\n".join([f"[{time_scale} PRIOR]: {text}" for time_scale, text in priors.items()])
        user_message = f"TEMPORAL CONTEXT:\n{context_str}\n\nLATEST NEWS (T-0):\n{current_news}\n\nRECONCILE AND SUGGEST STATE:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = self.client.chat(
                messages=messages,
                temperature=0.2,
                max_tokens=800,
            )

            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > 0:
                return json.loads(response[start_idx:end_idx])
            else:
                raise ValueError(f"No JSON found in LLM response.")
        except Exception as e:
            raise RuntimeError(f"Temporal manifold mapping failed: {e}")

    def _build_temporal_system_prompt(self) -> str:
        return """You are a Senior Quantitative Strategist. You must reconcile multi-scale temporal context to suggest a manifold state.

HIERARCHICAL RECONCILIATION RULES:
1. T-360 (Annual): Sets the 'Manifold Anchor'. If the anchor is 'Structural Deficit', Ricci Curvature must remain negative regardless of news.
2. T-7 (Weekly): Sets the 'Momentum Vector'. Defines the starting Rotor state.
3. Model Views: If other models report high Greeks (Vega/Gamma), this must be reflected in the KAN feature vector.
4. T-0 (News): Triggers the 'Instantaneous Shift' (Rotor Spike).

GEOMETRIC CONSTRAINTS:
- ricci_mean_core_60d: -50.0 (Extreme Crisis) to 5.0 (Predictable Growth).
- ga_rotor_magnitude: 0.0 (Stationary) to 1.5 (Total Regime Break).
- If Strategic View (T-360) and Tactical News (T-0) diverge, prioritize the Strategic View for Curvature and Tactical News for Rotor Magnitude.

OUTPUT FORMAT (JSON):
{
  "regime": "STABLE" | "TRANSITION" | "STRESS" | "RECOVERY",
  "ga_rotor_magnitude": float,
  "ricci_mean_core_60d": float,
  "realized_vol_20d": float,
  "contextual_calibration": "Explanation of how T-360 anchored the view while T-0 shocked it.",
  "greek_adjustment": "Directional bias for Delta/Vega based on prior reports."
}"""
