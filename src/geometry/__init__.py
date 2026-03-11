"""
Geometry and graph-topology feature extraction.

This package houses the Ricci-curvature / Ricci-flow based tooling as well as
geometric-algebra (Clifford) regime feature construction.

Note: some environments (e.g., Drive/FUSE mounts) can break Python's default
temporary directory probing, which in turn can prevent importing PyTorch.
We conditionally set a workspace tempdir here so that geometry modules that
import torch work reliably.
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
        tempfile.gettempdir()
        return
    except FileNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    tmp_dir = repo_root / ".tmp"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        tmp_dir = repo_root
    tempfile.tempdir = str(tmp_dir)
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))
    os.environ.setdefault("TMP", str(tmp_dir))


_configure_tempdir()

