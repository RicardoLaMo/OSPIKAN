"""
KAN Knowledge Store: neural network-based storage of regime dynamics.

Stores regime-conditional volatility surface, covariance matrix, and
regime transition probabilities as learned KAN networks.
"""

from .config import KANStoreConfig
from .store import KANKnowledgeStore

__all__ = [
    "KANStoreConfig",
    "KANKnowledgeStore",
]
