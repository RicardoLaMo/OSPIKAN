"""
Workspace-safe Python runtime tweaks.

This repo lives on a Google Drive / FUSE-like mount where exclusive-create
(`O_EXCL`) file operations can fail. Python's `tempfile.gettempdir()` probes
candidate directories using exclusive-create, which breaks imports for packages
that call it at import time (e.g., `torch` -> `dill`).

Setting `tempfile.tempdir` avoids the probe and keeps everything inside the
workspace.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _configure_tempdir() -> None:
    repo_root = Path(__file__).resolve().parent
    tmp_dir = repo_root / ".tmp"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If the directory can't be created, fall back to the current working dir.
        tmp_dir = Path.cwd()

    # Avoid `tempfile` probing logic (uses exclusive-create, which can fail on Drive mounts).
    tempfile.tempdir = str(tmp_dir)

    # Also set env vars for child processes / libraries that read these directly.
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))
    os.environ.setdefault("TMP", str(tmp_dir))


_configure_tempdir()
