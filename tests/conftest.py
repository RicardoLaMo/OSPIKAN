from __future__ import annotations

import os
import tempfile
from pathlib import Path

# PyTorch imports `dill`, which calls `tempfile.gettempdir()` at import time.
# On some Drive/FUSE mounts, Python's tempdir probing (exclusive-create) fails.
# Setting `tempfile.tempdir` avoids the probe and keeps tests self-contained.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TMP_DIR = _REPO_ROOT / ".tmp"
_TMP_DIR.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(_TMP_DIR)
os.environ.setdefault("TMPDIR", str(_TMP_DIR))
os.environ.setdefault("TEMP", str(_TMP_DIR))
os.environ.setdefault("TMP", str(_TMP_DIR))

