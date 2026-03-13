#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "silver" / "train_spikan_high_fidelity.py"),
    run_name="__main__",
)
