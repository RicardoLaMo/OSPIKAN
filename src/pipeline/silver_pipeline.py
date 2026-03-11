import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class UniverseConfig:
    provider: str
    start_date: str
    end_date: Optional[str]
    interval: str
    universe: Dict[str, List[str]]
    outputs: Dict[str, str]


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


# Allow `python src/pipeline/silver_pipeline.py ...` while still importing `src.*`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return slug.strip("._-") or "item"


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_universe_config(path: str) -> UniverseConfig:
    cfg = _read_yaml(path)

    defaults = cfg.get("defaults", {}) or {}
    outputs = cfg.get("outputs", {}) or {}
    universe = cfg.get("universe", {}) or {}

    provider = str(defaults.get("provider", "yfinance"))
    start_date = str(defaults.get("start_date", "2010-01-01"))
    end_date = defaults.get("end_date", None)
    interval = str(defaults.get("interval", "1d"))

    if not isinstance(universe, dict) or not universe:
        raise ValueError("Invalid config: `universe` must be a non-empty mapping of group -> symbols.")

    normalized_universe: Dict[str, List[str]] = {}
    for group, symbols in universe.items():
        if not symbols:
            continue
        if not isinstance(symbols, list) or not all(isinstance(s, str) and s.strip() for s in symbols):
            raise ValueError(f"Invalid config: universe group `{group}` must be a list of non-empty strings.")
        normalized_universe[str(group)] = [s.strip() for s in symbols]

    if not normalized_universe:
        raise ValueError("Invalid config: no symbols found under `universe`.")

    default_outputs = {
        "raw_dir": "data/raw",
        "interim_dir": "data/interim",
        "processed_dir": "data/processed",
        "metadata_dir": "data/raw/_runs",
    }
    merged_outputs = {**default_outputs, **{str(k): str(v) for k, v in outputs.items()}}

    return UniverseConfig(
        provider=provider,
        start_date=start_date,
        end_date=str(end_date) if end_date not in (None, "null") else None,
        interval=interval,
        universe=normalized_universe,
        outputs=merged_outputs,
    )


def flatten_universe(universe: Dict[str, List[str]]) -> List[str]:
    symbols: List[str] = []
    seen = set()
    for group in universe.values():
        for symbol in group:
            if symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
    return symbols


def _git_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
            .strip()
            .splitlines()[0]
        )
    except Exception:
        return None


def _git_is_dirty() -> Optional[bool]:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True)
        return bool(out.strip())
    except Exception:
        return None


def _make_run_id(config_path: str, symbols: List[str]) -> str:
    payload = {"config_path": config_path, "symbols": symbols}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{_utc_now_compact()}_{digest}"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_parquet_safe(df: pd.DataFrame, path: str) -> None:
    """
    Writes Parquet in a Drive-mount-safe way.

    Some FUSE/drive mounts can produce 0-byte Parquet files when pyarrow writes
    directly to the mount. Also, `tempfile.mkstemp()` uses exclusive-create
    semantics that can fail on these mounts.

    Strategy: serialize Parquet to an in-memory buffer, then write bytes with a
    regular file handle.
    """
    _ensure_dir(os.path.dirname(path) or ".")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError(
            "Parquet write requires `pyarrow`. Install it or switch outputs to CSV."
        ) from e

    table = pa.Table.from_pandas(df, preserve_index=True)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    buf = sink.getvalue().to_pybytes()

    with open(path, "wb") as f:
        f.write(buf)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize index to date-only, monotonic ascending, no duplicates.
    if df.index.name is None:
        df.index.name = "date"
    idx = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~idx.isna()].copy()
    df.index = idx[~idx.isna()].normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Normalize columns.
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=lower_map)

    rename_candidates = {
        "adj_close": "close_adj",
        "adjclose": "close_adj",
        "adjusted_close": "close_adj",
        "adjustedclose": "close_adj",
    }
    for src, dst in rename_candidates.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # Ensure required columns exist (as available).
    standard_cols = ["open", "high", "low", "close", "close_adj", "volume"]
    cols_present = [c for c in standard_cols if c in df.columns]
    df = df[cols_present]

    return df


def _fetch_openbb(symbol: str, start_date: str, end_date: Optional[str], provider: str) -> pd.DataFrame:
    from openbb import obb

    kwargs: Dict[str, Any] = {"start_date": start_date, "provider": provider}
    if end_date:
        kwargs["end_date"] = end_date

    # OpenBB endpoints are asset-class specific; equity is a pragmatic default for yfinance-backed symbols.
    return obb.equity.price.historical(symbol, **kwargs).to_df()


def _fetch_yfinance(symbol: str, start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        return df

    # yfinance returns standard OHLCV (Title Case) with an index name like 'Date'
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "close_adj",
            "Volume": "volume",
        }
    )
    return df


def fetch_symbol_history(
    symbol: str,
    *,
    start_date: str,
    end_date: Optional[str],
    provider: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns (normalized_df, metadata) for a single symbol.

    Metadata includes the provider path used and any exception notes.
    """
    meta: Dict[str, Any] = {"symbol": symbol, "provider": provider, "source": None, "error": None}

    try:
        raw = _fetch_openbb(symbol, start_date=start_date, end_date=end_date, provider=provider)
        meta["source"] = "openbb"
    except Exception as e:
        meta["error"] = f"openbb_failed: {e.__class__.__name__}: {e}"
        raw = pd.DataFrame()

    if raw is None or raw.empty:
        try:
            raw = _fetch_yfinance(symbol, start_date=start_date, end_date=end_date)
            meta["source"] = "yfinance"
            meta["error"] = None if not raw.empty else meta["error"]
        except Exception as e:
            meta["error"] = (meta["error"] or "") + f" | yfinance_failed: {e.__class__.__name__}: {e}"
            raw = pd.DataFrame()

    norm = _normalize_ohlcv(raw) if raw is not None and not raw.empty else pd.DataFrame()
    meta["rows"] = int(len(norm))
    meta["start"] = str(norm.index.min().date()) if len(norm) else None
    meta["end"] = str(norm.index.max().date()) if len(norm) else None
    meta["columns"] = list(norm.columns)
    return norm, meta


def ingest_universe(
    config_path: str,
    *,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_universe_config(config_path)
    symbols = flatten_universe(cfg.universe)

    run_id = run_id or _make_run_id(config_path, symbols)
    run_dir = os.path.join(cfg.outputs["metadata_dir"], run_id)
    _ensure_dir(run_dir)

    run_meta: Dict[str, Any] = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "provider": cfg.provider,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "interval": cfg.interval,
        "symbols": symbols,
        "git_sha": _git_sha(),
        "git_dirty": _git_is_dirty(),
        "results": {},
    }

    n_written = 0
    for symbol in symbols:
        df, meta = fetch_symbol_history(
            symbol,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            provider=cfg.provider,
        )

        symbol_slug = _safe_slug(symbol)
        out_path = os.path.join(run_dir, f"{symbol_slug}.csv")
        meta["output_path"] = out_path

        if not df.empty:
            df.to_csv(out_path, index=True)
            n_written += 1
        run_meta["results"][symbol] = meta

    # Write run metadata last so partial runs can be detected.
    meta_path = os.path.join(run_dir, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, sort_keys=True)

    if n_written == 0:
        example_errors: List[str] = []
        for sym, res in run_meta.get("results", {}).items():
            err = res.get("error")
            if err:
                example_errors.append(f"{sym}: {err}")

        msg = (
            "Ingest produced no data for any symbol. "
            f"Check dependencies/network and inspect run metadata at `{meta_path}`."
        )
        if example_errors:
            msg += " Example errors: " + " | ".join(example_errors[:3])
        raise RuntimeError(msg)

    with open(os.path.join(cfg.outputs["metadata_dir"], "LATEST"), "w", encoding="utf-8") as f:
        f.write(run_id + "\n")

    return {"run_id": run_id, "run_dir": run_dir, "symbols": symbols}


def build_interim_panel(
    run_dir: str,
    *,
    symbols: Iterable[str],
    how: str = "outer",
    ffill: bool = True,
) -> pd.DataFrame:
    """
    Builds a multi-asset close-price panel from a run directory.

    Output columns are the raw symbol strings (not slugs).
    """
    frames: List[pd.Series] = []
    for symbol in symbols:
        symbol_slug = _safe_slug(symbol)
        csv_path = os.path.join(run_dir, f"{symbol_slug}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if "close" not in df.columns:
            continue
        s = df["close"].copy()
        s.name = symbol
        frames.append(s)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, axis=1, join=how).sort_index()
    if ffill:
        panel = panel.ffill()
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Silver regime analysis pipeline (ingest + alignment).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Fetch and persist raw history for the configured universe.")
    p_ingest.add_argument("--config", default="configs/silver_universe.yaml")
    p_ingest.add_argument("--run-id", default=None)

    p_align = sub.add_parser("align", help="Build an aligned close-price panel from an ingest run.")
    p_align.add_argument("--config", default="configs/silver_universe.yaml")
    p_align.add_argument("--run-dir", default=None, help="Path like data/raw/_runs/<run_id>")
    p_align.add_argument("--run-id", default=None, help="If provided, resolves run dir under metadata_dir.")
    p_align.add_argument("--out", default=None, help="Output CSV path under data/interim/")

    p_qa = sub.add_parser("qa", help="Validate an aligned panel (missingness + basic sanity checks).")
    p_qa.add_argument("--panel", required=True, help="CSV path produced by `align` (date index in first column).")
    p_qa.add_argument("--out", default=None, help="Output JSON path (defaults next to panel).")
    p_qa.add_argument("--required", default=None, help="Comma-separated required column names.")
    p_qa.add_argument("--max-missing-rate", type=float, default=None, help="Fail if overall missing rate exceeds this.")

    p_feat = sub.add_parser("features", help="Compute a silver-focused feature table from an aligned close panel.")
    p_feat.add_argument("--panel", required=True, help="CSV/Parquet close panel (date index in first column).")
    p_feat.add_argument("--config", default=None, help="Config YAML with manifold definitions (optional, for sectional curvature).")
    p_feat.add_argument("--out", default=None, help="Output Parquet path (defaults under data/processed/).")
    p_feat.add_argument("--silver", default="SI=F", help="Silver symbol to use as target.")
    p_feat.add_argument("--gold", default="GC=F", help="Gold symbol to use for GSR.")
    p_feat.add_argument("--dxy", default="DX-Y.NYB", help="Dollar index symbol.")
    p_feat.add_argument("--y10", default="^TNX", help="10Y yield level symbol.")
    p_feat.add_argument("--spx", default="SPY", help="Equity proxy symbol.")
    p_feat.add_argument("--vix", default="^VIX", help="VIX symbol.")
    p_feat.add_argument("--deep-geometry", action="store_true",
                       help="Enable deep geometry analysis (Ricci flow stability). Increases runtime significantly.")
    p_feat.add_argument("--macro-features", action="store_true",
                       help="Enable enhanced macro features (treasury spreads, credit spreads, cross-asset ratios).")
    p_feat.add_argument("--sectional-curvature", action="store_true",
                       help="Enable sectional curvature (manifold-specific geometry). Requires --config with manifolds.")
    p_feat.add_argument("--fluid-dynamics", action="store_true",
                       help="Enable fluid dynamics features (shock formation index, momentum decay, viscosity).")

    args = parser.parse_args()

    if args.cmd == "ingest":
        out = ingest_universe(args.config, run_id=args.run_id)
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    if args.cmd == "align":
        cfg = load_universe_config(args.config)
        symbols = flatten_universe(cfg.universe)

        if args.run_dir:
            run_dir = args.run_dir
            run_id = os.path.basename(os.path.normpath(run_dir))
        else:
            if not args.run_id:
                latest_path = os.path.join(cfg.outputs["metadata_dir"], "LATEST")
                if not os.path.exists(latest_path):
                    raise FileNotFoundError("No --run-id provided and LATEST not found; run `ingest` first.")
                with open(latest_path, "r", encoding="utf-8") as f:
                    run_id = f.read().strip()
            else:
                run_id = args.run_id
            run_dir = os.path.join(cfg.outputs["metadata_dir"], run_id)

        panel = build_interim_panel(run_dir, symbols=symbols)
        if panel.empty:
            msg = "No data found to build interim panel (check run dir + symbol outputs)."
            meta_path = os.path.join(run_dir, "run_metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f) or {}
                    results = meta.get("results", {}) or {}
                    total = len(results)
                    nonzero = sum(1 for r in results.values() if int(r.get("rows") or 0) > 0)
                    errors = [r.get("error") for r in results.values() if r.get("error")]
                    unique_errors: List[str] = []
                    for e in errors:
                        if e and e not in unique_errors:
                            unique_errors.append(str(e))
                    msg += f" Run `{meta.get('run_id', os.path.basename(run_dir))}` produced {nonzero}/{total} non-empty series."
                    msg += f" Inspect `{meta_path}`."
                    if unique_errors:
                        msg += " Example error(s): " + " | ".join(unique_errors[:3])
                except Exception:
                    msg += f" (Could not parse `{meta_path}` for diagnostics.)"

            raise RuntimeError(msg)

        out_path = args.out or os.path.join(cfg.outputs["interim_dir"], f"silver_panel_close_{run_id}.csv")
        _ensure_dir(os.path.dirname(out_path))
        panel.to_csv(out_path, index=True)
        print(out_path)
        return

    if args.cmd == "qa":
        from src.validation.data_quality import validate_panel, write_json

        panel = pd.read_csv(args.panel, index_col=0, parse_dates=True)
        required = [c.strip() for c in (args.required or "").split(",") if c.strip()] or None

        report = validate_panel(panel, required_columns=required, max_missing_rate=args.max_missing_rate)

        out_path = args.out or (args.panel + ".qa.json")
        _ensure_dir(os.path.dirname(out_path) or ".")
        write_json(out_path, report)
        print(out_path)
        return

    if args.cmd == "features":
        import re
        from src.analysis.features import FeatureConfig, compute_silver_features, feature_metadata

        if args.panel.endswith(".parquet"):
            panel = pd.read_parquet(args.panel)
        else:
            panel = pd.read_csv(args.panel, index_col=0, parse_dates=True)

        run_id = None
        m = re.search(r"silver_panel_close_(.+)\.(csv|parquet)$", os.path.basename(args.panel))
        if m:
            run_id = m.group(1)
        else:
            run_id = _utc_now_compact()

        # Load manifolds from config if requested
        manifolds = None
        if hasattr(args, 'sectional_curvature') and args.sectional_curvature:
            if args.config:
                # Load manifolds directly from YAML (not part of UniverseConfig dataclass)
                config_yaml = _read_yaml(args.config)
                manifolds_cfg = config_yaml.get('manifolds', {})
                if manifolds_cfg:
                    # Convert manifold config to dict of symbol lists
                    manifolds = {}
                    for name, mani_spec in manifolds_cfg.items():
                        if isinstance(mani_spec, dict) and 'symbols' in mani_spec:
                            manifolds[name] = mani_spec['symbols']
                        elif isinstance(mani_spec, list):
                            # Direct list of symbols
                            manifolds[name] = mani_spec
                    print(f"Loaded {len(manifolds)} manifolds from config for sectional curvature", file=sys.stderr)
                else:
                    print("Warning: --sectional-curvature requested but no manifolds in config", file=sys.stderr)
            else:
                print("Warning: --sectional-curvature requested but no --config provided", file=sys.stderr)

        cfg = FeatureConfig()
        features = compute_silver_features(
            panel,
            silver_symbol=args.silver,
            gold_symbol=args.gold or None,
            dxy_symbol=args.dxy or None,
            y10_symbol=args.y10 or None,
            spx_symbol=args.spx or None,
            vix_symbol=args.vix or None,
            config=cfg,
            include_macro_features=hasattr(args, 'macro_features') and args.macro_features,
            include_fluid_dynamics=hasattr(args, 'fluid_dynamics') and args.fluid_dynamics,
            manifolds=manifolds,
        )

        # Add Geometric Algebra regime features (requires base features to exist)
        print("Computing GA regime features (Cl(4,0) rotors)...", file=sys.stderr)
        from src.geometry.ga_regime_features import compute_ga_regime_features
        ga_features = compute_ga_regime_features(features, window=60, device="cpu")
        features = pd.concat([features, ga_features], axis=1)

        # Optional: Deep geometry analysis (Ricci flow stability)
        if hasattr(args, 'deep_geometry') and args.deep_geometry:
            print("Computing Ricci flow stability (deep geometry mode - may take several minutes)...", file=sys.stderr)
            from src.geometry.ricci_flow import rolling_ricci_flow_stability
            returns = panel.apply(lambda x: np.log(x / x.shift(1)), axis=0).dropna(axis=1, how="all")
            if returns.shape[1] >= 3:
                flow_stability = rolling_ricci_flow_stability(returns, window=60, flow_steps=20, min_assets=3)
                features["ricci_flow_stability_60d"] = flow_stability
                print(f"  Ricci flow stability computed: {flow_stability.notna().sum()} valid values", file=sys.stderr)
            else:
                print("  Skipped: insufficient assets for flow analysis", file=sys.stderr)

        out_path = args.out or os.path.join("data", "processed", f"silver_features_{run_id}.parquet")
        _ensure_dir(os.path.dirname(out_path))
        _write_parquet_safe(features, out_path)

        meta_path = out_path.replace(".parquet", ".metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(feature_metadata(
                silver_symbol=args.silver,
                gold_symbol=args.gold or None,
                dxy_symbol=args.dxy or None,
                y10_symbol=args.y10 or None,
                spx_symbol=args.spx or None,
                vix_symbol=args.vix or None,
                config=cfg,
                include_macro_features=hasattr(args, 'macro_features') and args.macro_features,
                include_fluid_dynamics=hasattr(args, 'fluid_dynamics') and args.fluid_dynamics,
                manifolds=manifolds,
            ), f, indent=2, sort_keys=True)

        print(out_path)
        return


if __name__ == "__main__":
    main()
