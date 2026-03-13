#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "silver" / "run_out_of_sample_validation.py"),
    run_name="__main__",
)
