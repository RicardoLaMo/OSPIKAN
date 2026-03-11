"""
Physics-Informed Neural Networks for Market Fluid Dynamics.

This module implements SPIKAN (Separable Physics-Informed Kolmogorov-Arnold Networks)
for modeling market dynamics as fluid systems. Key components:

- kan_layers: B-spline based KAN layers (learnable activation functions)
- spikan: Separable PIKAN architecture for efficient multivariate function approximation
- market_pdes: Market-specific PDE definitions (Burgers, advection-diffusion)
- shock_detector: Discontinuity and shock detection in price fields

The physics module complements the geometry module:
- Ricci curvature (geometry): Static market structure, topology, bottlenecks
- SPIKAN PDEs (physics): Dynamic evolution, shock propagation, momentum decay
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _configure_tempdir() -> None:
    """
    Ensure `tempfile.gettempdir()` works on Drive/FUSE mounts.

    Some mounts fail Python's default tempdir probing (uses exclusive-create).
    We set `tempfile.tempdir` to a workspace directory to avoid the probe.
    """
    try:
        # If the system tempdir is usable, don't override it (important for
        # torch.compile / inductor caches and general OS integration).
        tempfile.gettempdir()
        return
    except FileNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    tmp_dir = repo_root / ".tmp"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fall back to repo root if we cannot create the temp directory.
        tmp_dir = repo_root
    tempfile.tempdir = str(tmp_dir)
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))
    os.environ.setdefault("TMP", str(tmp_dir))


_configure_tempdir()

from .kan_layers import KANLayer, KANLinear, bspline_basis
from .spikan import MarketSPIKAN, SPIKANConfig
from .market_pdes import BurgersEquation, AdvectionDiffusion, physics_informed_loss

__all__ = [
    "KANLayer",
    "KANLinear",
    "bspline_basis",
    "MarketSPIKAN",
    "SPIKANConfig",
    "BurgersEquation",
    "AdvectionDiffusion",
    "physics_informed_loss",
]
