import os
import sys

import pandas as pd
import pytest

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.silver_pipeline import build_interim_panel, flatten_universe, load_universe_config
import src.pipeline.silver_pipeline as silver_pipeline


def test_load_universe_config_roundtrip(tmp_path):
    config_path = tmp_path / "silver_universe.yaml"
    config_path.write_text(
        """
version: 1
defaults:
  provider: yfinance
  start_date: "2020-01-01"
  end_date: null
  interval: 1d
universe:
  silver:
    - SI=F
    - SLV
  macro:
    - ^TNX
outputs:
  metadata_dir: data/raw/_runs
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_universe_config(str(config_path))

    assert cfg.provider == "yfinance"
    assert cfg.start_date == "2020-01-01"
    assert cfg.end_date is None
    assert cfg.interval == "1d"
    assert set(cfg.universe.keys()) == {"silver", "macro"}
    assert cfg.outputs["metadata_dir"] == "data/raw/_runs"


def test_flatten_universe_dedup_order():
    universe = {"a": ["X", "Y"], "b": ["Y", "Z"]}
    symbols = flatten_universe(universe)
    assert symbols == ["X", "Y", "Z"]


def test_build_interim_panel_reads_close(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Two series with a gap to force alignment + ffill.
    pd.DataFrame(
        {"close": [10.0, 11.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-03"]),
    ).to_csv(run_dir / "AAA.csv")

    pd.DataFrame(
        {"close": [20.0, 21.0, 22.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    ).to_csv(run_dir / "BBB.csv")

    panel = build_interim_panel(str(run_dir), symbols=["AAA", "BBB"])

    assert list(panel.columns) == ["AAA", "BBB"]
    assert panel.loc["2024-01-02", "AAA"] == 10.0
    assert panel.loc["2024-01-02", "BBB"] == 21.0


def test_ingest_universe_raises_when_all_symbols_empty(tmp_path, monkeypatch):
    config_path = tmp_path / "silver_universe.yaml"
    metadata_dir = tmp_path / "runs"

    config_path.write_text(
        f"""
version: 1
defaults:
  provider: yfinance
  start_date: "2020-01-01"
  end_date: null
  interval: 1d
universe:
  silver:
    - SI=F
outputs:
  metadata_dir: {metadata_dir.as_posix()}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def fake_fetch_symbol_history(symbol: str, *, start_date: str, end_date: str | None, provider: str):
        df = pd.DataFrame()
        meta = {
            "symbol": symbol,
            "provider": provider,
            "source": None,
            "error": "test_empty",
            "rows": 0,
            "start": None,
            "end": None,
            "columns": [],
        }
        return df, meta

    monkeypatch.setattr(silver_pipeline, "fetch_symbol_history", fake_fetch_symbol_history)

    with pytest.raises(RuntimeError):
        silver_pipeline.ingest_universe(str(config_path))

    assert metadata_dir.exists()
    run_dirs = [p for p in metadata_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "run_metadata.json").exists()
    assert not (metadata_dir / "LATEST").exists()
