"""
Enhanced macro feature engineering for silver analysis.

Computes orthogonal macro signals:
- Treasury spreads (yield curve)
- Credit spreads (IG-HY, EM-DM)
- Cross-asset ratios (gold/oil, copper/gold, equity/bond)
- Volatility metrics (VIX structure, equity-bond vol ratio)

These features are orthogonal to simple price-based metrics and capture
regime-specific risk factors (credit cycle, term premium, risk appetite).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _safe_log_return(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Log return that is robust to non-positive prices (e.g., WTI went negative in 2020).

    Returns NaN when current or lagged price is <= 0.
    """
    s = series.astype(float)
    prev = s.shift(int(periods))
    valid = (s > 0) & (prev > 0)
    out = pd.Series(index=s.index, dtype="float64")
    out.loc[valid] = np.log(s.loc[valid] / prev.loc[valid])
    return out.replace([np.inf, -np.inf], np.nan)


def compute_treasury_spreads(
    panel: pd.DataFrame,
    *,
    y3m_symbol: str = "^IRX",
    y2_symbol: Optional[str] = None,
    y5_symbol: str = "^FVX",
    y10_symbol: str = "^TNX",
    y30_symbol: str = "^TYX",
) -> pd.DataFrame:
    """
    Computes treasury yield spreads (yield curve indicators).

    Key spreads:
    - slope_2y_10y: Classic yield curve slope (recession indicator)
    - slope_10y_30y: Long-end steepening (growth expectations)
    - term_premium: 10Y-3M spread (term premium, liquidity preference)

    Args:
        panel: Close price panel with treasury yield symbols
        y3m_symbol: 3-month T-Bill yield symbol (^IRX)
        y2_symbol: 2-year Treasury yield symbol (often not available)
        y5_symbol: 5-year Treasury yield symbol (^FVX)
        y10_symbol: 10-year Treasury yield symbol (^TNX)
        y30_symbol: 30-year Treasury yield symbol (^TYX)

    Returns:
        DataFrame with treasury spread features
    """
    out = pd.DataFrame(index=panel.index)

    # Extract yields (these are already in % terms from Yahoo)
    y3m = panel[y3m_symbol].astype(float) if y3m_symbol in panel.columns else None
    y5 = panel[y5_symbol].astype(float) if y5_symbol in panel.columns else None
    y10 = panel[y10_symbol].astype(float) if y10_symbol in panel.columns else None
    y30 = panel[y30_symbol].astype(float) if y30_symbol in panel.columns else None

    # Yield curve slope (2Y-10Y proxy: use 5Y if 2Y not available)
    if y10 is not None and y5 is not None:
        out["slope_5y_10y"] = y10 - y5  # Positive = normal curve, negative = inverted

    # Long-end steepness (10Y-30Y)
    if y30 is not None and y10 is not None:
        out["slope_10y_30y"] = y30 - y10  # Positive = steep long end (growth)

    # Term premium (10Y-3M)
    if y10 is not None and y3m is not None:
        out["term_premium"] = y10 - y3m  # Positive = normal, negative = inversion (recession)

    # Yield curve level (average of available yields)
    available_yields = [y for y in [y3m, y5, y10, y30] if y is not None]
    if available_yields:
        out["yield_curve_level"] = pd.concat(available_yields, axis=1).mean(axis=1)

    # Yield curve changes (momentum in rates)
    if "term_premium" in out.columns:
        out["term_premium_change_20d"] = out["term_premium"].diff(20)

    return out


def compute_credit_spreads(
    panel: pd.DataFrame,
    *,
    hyg_symbol: str = "HYG",
    lqd_symbol: str = "LQD",
    jnk_symbol: Optional[str] = "JNK",
    emb_symbol: Optional[str] = "EMB",
    ief_symbol: str = "IEF",  # 7-10Y Treasury ETF (risk-free proxy)
) -> pd.DataFrame:
    """
    Computes credit spreads from bond ETF prices.

    Note: ETF prices are inverse to yields (price up = yield down).
    Credit spread = yield_risky - yield_safe.
    Approximation: Use log(safe_price / risky_price) as spread proxy.

    Key spreads:
    - ig_hy_spread: Investment Grade vs High Yield (credit cycle)
    - hy_treasury_spread: High Yield vs Treasury (credit risk premium)
    - em_ig_spread: Emerging Market vs IG (EM risk premium)

    Args:
        panel: Close price panel with bond ETF symbols
        hyg_symbol: High Yield Corporate Bond ETF (HYG)
        lqd_symbol: Investment Grade Corporate Bond ETF (LQD)
        jnk_symbol: Alternative High Yield ETF (JNK)
        emb_symbol: Emerging Market Bond ETF (EMB)
        ief_symbol: Treasury ETF for risk-free rate (IEF)

    Returns:
        DataFrame with credit spread features
    """
    out = pd.DataFrame(index=panel.index)

    # Extract bond ETF prices
    hyg = panel[hyg_symbol].astype(float) if hyg_symbol in panel.columns else None
    lqd = panel[lqd_symbol].astype(float) if lqd_symbol in panel.columns else None
    jnk = panel[jnk_symbol].astype(float) if jnk_symbol and jnk_symbol in panel.columns else None
    emb = panel[emb_symbol].astype(float) if emb_symbol and emb_symbol in panel.columns else None
    ief = panel[ief_symbol].astype(float) if ief_symbol in panel.columns else None

    # IG-HY spread (credit cycle indicator)
    # When spread widens (LQD outperforms HYG): Credit stress, flight to quality
    # When spread tightens (HYG outperforms LQD): Risk-on, credit improving
    if lqd is not None and hyg is not None:
        # Spread proxy: log(LQD/HYG) - higher = wider spread = more stress
        out["ig_hy_spread"] = np.log(lqd / hyg)

        # Spread change (momentum in credit conditions)
        out["ig_hy_spread_change_20d"] = out["ig_hy_spread"].diff(20)

        # Z-scored spread (relative to history)
        spread_mean = out["ig_hy_spread"].rolling(252).mean()
        spread_std = out["ig_hy_spread"].rolling(252).std() + 1e-8
        out["ig_hy_spread_zscore"] = (out["ig_hy_spread"] - spread_mean) / spread_std

    # HY-Treasury spread (credit risk premium)
    if ief is not None and hyg is not None:
        out["hy_treasury_spread"] = np.log(ief / hyg)

    # EM-IG spread (emerging market risk premium)
    if emb is not None and lqd is not None:
        out["em_ig_spread"] = np.log(lqd / emb)

    # Alternative HY spread (JNK vs LQD) for robustness
    if jnk is not None and lqd is not None:
        out["jnk_lqd_spread"] = np.log(lqd / jnk)

    # Credit beta to equities (does HYG move with SPY? If yes, risk-on)
    if hyg is not None and "SPY" in panel.columns:
        spy = panel["SPY"].astype(float)
        hyg_ret = np.log(hyg / hyg.shift(1))
        spy_ret = np.log(spy / spy.shift(1))

        # Rolling beta: HYG returns ~ SPY returns
        cov = hyg_ret.rolling(60).cov(spy_ret)
        var = spy_ret.rolling(60).var()
        out["hyg_equity_beta_60d"] = cov / var

    return out


def compute_cross_asset_ratios(
    panel: pd.DataFrame,
    *,
    gold_symbol: str = "GC=F",
    silver_symbol: str = "SI=F",
    copper_symbol: str = "HG=F",
    oil_symbol: str = "CL=F",
    spy_symbol: str = "SPY",
    tlt_symbol: str = "TLT",
    eem_symbol: Optional[str] = "EEM",
    efa_symbol: Optional[str] = "EFA",
) -> pd.DataFrame:
    """
    Computes cross-asset ratios (orthogonal regime indicators).

    These ratios capture regime transitions that single-asset analysis misses:
    - Gold/Oil: Monetary vs energy (>25 = monetary dominance)
    - Copper/Gold: Growth vs haven (>0.05 = growth dominance, Dr. Copper)
    - Equity/Bond: Risk-on vs risk-off (rising = risk-on)
    - EM/DM: Emerging market risk appetite

    Args:
        panel: Close price panel with asset symbols

    Returns:
        DataFrame with cross-asset ratio features
    """
    out = pd.DataFrame(index=panel.index)

    # Extract prices
    gold = panel[gold_symbol].astype(float) if gold_symbol in panel.columns else None
    silver = panel[silver_symbol].astype(float) if silver_symbol in panel.columns else None
    copper = panel[copper_symbol].astype(float) if copper_symbol in panel.columns else None
    oil = panel[oil_symbol].astype(float) if oil_symbol in panel.columns else None
    spy = panel[spy_symbol].astype(float) if spy_symbol in panel.columns else None
    tlt = panel[tlt_symbol].astype(float) if tlt_symbol in panel.columns else None
    eem = panel[eem_symbol].astype(float) if eem_symbol and eem_symbol in panel.columns else None
    efa = panel[efa_symbol].astype(float) if efa_symbol and efa_symbol in panel.columns else None

    # Gold/Oil ratio (monetary vs energy regime)
    # Historical range: 15-30
    # >25 = monetary asset dominance (deflation fear, energy weakness)
    # <20 = energy dominance (inflation, growth)
    if gold is not None and oil is not None:
        out["gold_oil_ratio"] = gold / oil
        out["gold_oil_ratio_zscore"] = (
            (out["gold_oil_ratio"] - out["gold_oil_ratio"].rolling(252).mean())
            / (out["gold_oil_ratio"].rolling(252).std() + 1e-8)
        )

    # Copper/Gold ratio (Dr. Copper - growth vs haven)
    # Rising = growth expectations (copper industrial demand)
    # Falling = haven demand (gold flight to safety)
    if copper is not None and gold is not None:
        out["copper_gold_ratio"] = copper / gold
        out["copper_gold_ratio_zscore"] = (
            (out["copper_gold_ratio"] - out["copper_gold_ratio"].rolling(252).mean())
            / (out["copper_gold_ratio"].rolling(252).std() + 1e-8)
        )

    # Equity/Bond ratio (risk-on vs risk-off)
    # Rising = risk-on (equities outperform bonds)
    # Falling = risk-off (bonds outperform equities)
    if spy is not None and tlt is not None:
        out["equity_bond_ratio"] = spy / tlt
        out["equity_bond_ratio_change_20d"] = out["equity_bond_ratio"].pct_change(20)

    # EM/DM equity ratio (emerging market risk appetite)
    # Rising = EM outperforming (risk appetite high)
    # Falling = EM underperforming (risk aversion)
    if eem is not None and efa is not None:
        out["em_dm_equity_ratio"] = eem / efa
        out["em_dm_ratio_zscore"] = (
            (out["em_dm_equity_ratio"] - out["em_dm_equity_ratio"].rolling(252).mean())
            / (out["em_dm_equity_ratio"].rolling(252).std() + 1e-8)
        )

    # Silver/Gold ratio momentum (GSR itself already in base features)
    # Note: Base features compute 'gsr' = gold/silver, we add momentum here
    if silver is not None and gold is not None:
        gsr = gold / silver
        out["gsr_momentum_20d"] = gsr.pct_change(20)

        # Z-scored GSR for regime identification
        gsr_mean = gsr.rolling(252).mean()
        gsr_std = gsr.rolling(252).std() + 1e-8
        out["gsr_zscore"] = (gsr - gsr_mean) / gsr_std

    return out


def compute_volatility_metrics(
    panel: pd.DataFrame,
    *,
    vix_symbol: str = "^VIX",
    vxn_symbol: Optional[str] = "^VXN",
    tlt_symbol: str = "TLT",
    spy_symbol: str = "SPY",
) -> pd.DataFrame:
    """
    Computes volatility-based regime indicators.

    Key metrics:
    - VIX level (equity vol)
    - VIX term structure slope (VXN/VIX proxy)
    - Equity-bond vol ratio (flight to quality indicator)
    - Vol-of-vol (tail risk)

    Args:
        panel: Close price panel with volatility and price symbols

    Returns:
        DataFrame with volatility features
    """
    out = pd.DataFrame(index=panel.index)

    # VIX level (skip if already in base features, but add regime thresholds)
    vix = panel[vix_symbol].astype(float) if vix_symbol in panel.columns else None
    if vix is not None:
        # Don't duplicate vix_level (already in base features)
        # out["vix_level"] = vix  # REMOVED - already in base features

        # VIX regime flags (new)
        out["vix_high"] = (vix > 25).astype(int)  # Stress regime
        out["vix_extreme"] = (vix > 35).astype(int)  # Crisis regime

        # VIX momentum (rising vol = increasing fear)
        out["vix_change_5d"] = vix.diff(5)
        out["vix_change_20d"] = vix.diff(20)

    # VIX term structure slope proxy (VXN/VIX)
    # Contango (VXN > VIX) = calm markets, vol expected to decline
    # Backwardation (VXN < VIX) = stress, vol expected to persist
    vxn = panel[vxn_symbol].astype(float) if vxn_symbol and vxn_symbol in panel.columns else None
    if vix is not None and vxn is not None:
        out["vix_term_slope"] = (vxn - vix) / vix  # Normalized slope

    # Equity-bond volatility ratio (flight to quality)
    # High = equity vol >> bond vol (panic into bonds)
    # Low = equity vol ~ bond vol (normal regime)
    spy = panel[spy_symbol].astype(float) if spy_symbol in panel.columns else None
    tlt = panel[tlt_symbol].astype(float) if tlt_symbol in panel.columns else None

    if spy is not None and tlt is not None:
        spy_ret = np.log(spy / spy.shift(1))
        tlt_ret = np.log(tlt / tlt.shift(1))

        spy_vol = spy_ret.rolling(20).std() * np.sqrt(252)
        tlt_vol = tlt_ret.rolling(20).std() * np.sqrt(252)

        out["equity_vol_20d"] = spy_vol
        out["bond_vol_20d"] = tlt_vol
        out["equity_bond_vol_ratio"] = spy_vol / (tlt_vol + 1e-8)

    # Vol-of-vol (tail risk measure)
    if vix is not None:
        vix_ret = np.log(vix / vix.shift(1))
        out["vix_vol_20d"] = vix_ret.rolling(20).std() * np.sqrt(252)

    return out


def compute_energy_features(
    panel: pd.DataFrame,
    *,
    wti_symbol: str = "CL=F",
    brent_symbol: Optional[str] = "BZ=F",
    natgas_symbol: Optional[str] = "NG=F",
    xle_symbol: Optional[str] = "XLE",
) -> pd.DataFrame:
    """
    Computes energy-specific features.

    Energy impacts silver through:
    - Mining costs (40% of silver is byproduct of base metal mining)
    - Industrial activity correlation (oil demand ~ manufacturing)
    - Inflation expectations (energy is CPI driver)

    Args:
        panel: Close price panel with energy symbols

    Returns:
        DataFrame with energy features
    """
    out = pd.DataFrame(index=panel.index)

    wti = panel[wti_symbol].astype(float) if wti_symbol in panel.columns else None
    brent = panel[brent_symbol].astype(float) if brent_symbol and brent_symbol in panel.columns else None
    natgas = panel[natgas_symbol].astype(float) if natgas_symbol and natgas_symbol in panel.columns else None
    xle = panel[xle_symbol].astype(float) if xle_symbol and xle_symbol in panel.columns else None

    # Oil price level and momentum
    if wti is not None:
        out["oil_level"] = wti
        out["oil_return_20d"] = _safe_log_return(wti, 20)
        out["oil_vol_60d"] = _safe_log_return(wti, 1).rolling(60).std() * np.sqrt(252)

    # Brent-WTI spread (oil market stress indicator)
    if brent is not None and wti is not None:
        out["brent_wti_spread"] = brent - wti

    # Natural gas (alternative energy, diversification)
    if natgas is not None:
        out["natgas_level"] = natgas

    # Energy sector equities (leading indicator for oil)
    if xle is not None:
        out["energy_sector_level"] = xle
        out["energy_sector_return_20d"] = np.log(xle / xle.shift(20))

    return out


def compute_all_macro_features(
    panel: pd.DataFrame,
    config: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Computes all enhanced macro features.

    Orchestrates computation of:
    - Treasury spreads
    - Credit spreads
    - Cross-asset ratios
    - Volatility metrics
    - Energy features

    Args:
        panel: Close price panel with all asset symbols
        config: Optional config dict with symbol overrides

    Returns:
        DataFrame with all macro features
    """
    config = config or {}

    features = pd.DataFrame(index=panel.index)

    # Treasury spreads
    treasury_features = compute_treasury_spreads(panel)
    features = pd.concat([features, treasury_features], axis=1)

    # Credit spreads
    credit_features = compute_credit_spreads(panel)
    features = pd.concat([features, credit_features], axis=1)

    # Cross-asset ratios
    ratio_features = compute_cross_asset_ratios(panel)
    features = pd.concat([features, ratio_features], axis=1)

    # Volatility metrics
    vol_features = compute_volatility_metrics(panel)
    features = pd.concat([features, vol_features], axis=1)

    # Energy features
    energy_features = compute_energy_features(panel)
    features = pd.concat([features, energy_features], axis=1)

    return features
