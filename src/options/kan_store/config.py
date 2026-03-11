"""
Configuration for KAN Knowledge Store.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class KANStoreConfig:
    """Configuration for KAN Knowledge Store."""

    # Architecture
    vol_surface_layers: List[int] = None  # default: [10, 32, 32, 16, 1]
    covariance_layers: List[int] = None   # default: [8, 32, 32, 21]
    transition_layers: List[int] = None   # default: [9, 32, 32, 4]

    # KAN layer configuration
    grid_size: int = 8
    spline_order: int = 3
    grid_range: tuple = (-1.0, 1.0)
    base_activation: str = "silu"
    grid_eps: float = 0.02

    # Training
    dropout: float = 0.1
    residual: bool = True

    def __post_init__(self):
        """Set default layer sizes if not provided."""
        if self.vol_surface_layers is None:
            self.vol_surface_layers = [10, 32, 32, 16, 1]
        if self.covariance_layers is None:
            self.covariance_layers = [8, 32, 32, 21]
        if self.transition_layers is None:
            self.transition_layers = [9, 32, 32, 4]
