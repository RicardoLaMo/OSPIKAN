"""
Gromov-Wasserstein Optimal Transport for Manifold Validation.

Measures relational distances between disparate market manifolds to validate
if predicted geometric states are isometrically consistent with historical ground truth.
"""

import numpy as np
import ot
from typing import Tuple, Optional


class GromovWassersteinValidator:
    """Validates manifold consistency using Gromov-Wasserstein Optimal Transport."""

    def __init__(self, epsilon: float = 0.05, max_iter: int = 50):
        """
        Initialize GW validator.

        Args:
            epsilon: Entropic regularization parameter (Sinkhorn)
            max_iter: Maximum iterations for the GW solver
        """
        self.epsilon = epsilon
        self.max_iter = max_iter

    def compute_relational_distance(
        self, 
        source_points: np.ndarray, 
        target_points: np.ndarray
    ) -> float:
        """
        Compute the Gromov-Wasserstein distance between two point clouds.
        
        This measures how much the internal metric (pairwise distances) of the 
        source manifold must be distorted to match the target manifold.

        Args:
            source_points: (N, D1) array (e.g., KAN predicted surface)
            target_points: (M, D2) array (e.g., Historical crisis cloud)

        Returns:
            GW distance (float)
        """
        # 1. Compute intra-manifold distance matrices (cost matrices)
        C1 = ot.dist(source_points, source_points, metric='sqeuclidean')
        C2 = ot.dist(target_points, target_points, metric='sqeuclidean')

        # 2. Normalize matrices to preserve scale-invariance
        C1 /= C1.max() if C1.max() > 0 else 1.0
        C2 /= C2.max() if C2.max() > 0 else 1.0

        # 3. Uniform probability distributions over points
        p = ot.unif(len(source_points))
        q = ot.unif(len(target_points))

        # 4. Solve Entropic Gromov-Wasserstein
        # Returns: coupling matrix T
        T = ot.gromov.entropic_gromov_wasserstein(
            C1, C2, p, q, loss_fun='square_loss', epsilon=self.epsilon, 
            max_iter=self.max_iter
        )

        # 5. Compute the GW distance from the coupling matrix
        gw_dist = ot.gromov.gromov_wasserstein2(
            C1, C2, p, q, loss_fun='square_loss'
        )

        return float(gw_dist)

    def validate_isometry(
        self, 
        prediction: np.ndarray, 
        historical_reference: np.ndarray,
        threshold: float = 0.1
    ) -> Tuple[bool, float]:
        """
        Validate if a prediction is isometrically consistent with history.

        Args:
            prediction: The predicted manifold state
            historical_reference: Ground truth historical data
            threshold: Maximum allowed GW distance for 'High Fidelity'

        Returns:
            (is_valid, distance)
        """
        dist = self.compute_relational_distance(prediction, historical_reference)
        is_valid = dist < threshold
        return is_valid, dist
