import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
        run_meta["results"][symbol] = meta

    # Write run metadata last so partial runs can be detected.
    with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, sort_keys=True)

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
            raise RuntimeError("No data found to build interim panel (check run dir + symbol outputs).")

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


if __name__ == "__main__":
    main()
