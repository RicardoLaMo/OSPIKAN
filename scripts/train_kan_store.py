#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "options" / "train_kan_store.py"),
    run_name="__main__",
)
