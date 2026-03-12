"""
Abstract Syntax Tree (AST) node definitions for the options pricing DSL.

Each node represents a top-level query or command in the DSL.
"""

from dataclasses import dataclass
from typing import Optional, List, Union


@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    pass


@dataclass
class PriceQuery(ASTNode):
    """
    PRICE option type=call S=30 K=32 T=45d sigma=kan_regime [regime=STRESS] [r=0.05]

    Prices a European vanilla option using Black-Scholes with regime-adjusted volatility.
    """
    option_type: str  # 'call' or 'put'
    spot_price: float  # S
    strike_price: float  # K
    time_to_expiry: float  # T in years (e.g., 45/365 for 45 days)
    volatility: Union[float, str]  # float or 'kan_regime' or 'hist_Nd'
    regime: Optional[str] = None  # 'STABLE', 'TRANSITION', 'STRESS', 'RECOVERY', 'current'
    risk_free_rate: float = 0.05  # r
    dividend_yield: float = 0.0  # q


@dataclass
class RegimeCurrentQuery(ASTNode):
    """
    REGIME current asset=silver

    Queries current regime classification for a given asset.
    """
    asset: str  # 'silver', 'gold', 'dxy', etc.


@dataclass
class RegimeProbQuery(ASTNode):
    """
    REGIME prob from=STABLE to=STRESS horizon=10d

    Queries transition probability from one regime to another over a horizon.
    """
    from_regime: str  # source regime
    to_regime: str  # target regime
    horizon: float  # in years (e.g., 10/365 for 10 days)


@dataclass
class CovarianceQuery(ASTNode):
    """
    COVARIANCE assets=[silver,gold,dxy] regime=STRESS window=60d

    Returns covariance matrix for specified assets in a given regime.
    """
    assets: List[str]  # list of asset names
    regime: str  # regime name
    window: float  # in years (lookback window)


@dataclass
class TransitionQuery(ASTNode):
    """
    TRANSITION matrix asset=silver normalize=true

    Returns transition matrix for regime changes of a given asset.
    """
    asset: str
    normalize: bool = True


@dataclass
class WhatIfQuery(ASTNode):
    """
    WHAT_IF regime_shift to=STRESS asset=silver show=[delta,vega,vol,price]

    Analyzes how option Greeks change with regime shift.
    """
    to_regime: str  # target regime
    asset: str  # asset name
    spot_price: Optional[float] = None  # optional override
    strike_price: Optional[float] = None  # optional override
    time_to_expiry: Optional[float] = None  # optional override
    show: List[str] = None  # ['delta', 'vega', 'vol', 'price', 'gamma', 'theta']


@dataclass
class ExplainQuery(ASTNode):
    """
    EXPLAIN regime=TRANSITION features=[ricci_curvature,mst_stress,ga_rotor]

    Explains characteristics of a regime using specified features.
    """
    regime: str  # regime name
    features: List[str]  # feature names to explain


@dataclass
class SurfaceQuery(ASTNode):
    """
    SURFACE vol asset=silver regime=all strikes=[0.9,0.95,1.0,1.05,1.1] T=30d

    Generates vol surface across strikes and/or time.
    """
    asset: str
    regime: str  # 'all' or specific regime
    strikes: List[float]  # moneyness ratios or absolute strikes
    time_to_expiry: float  # in years
    vol_type: str = "implied"  # 'implied', 'realized', 'kan'


@dataclass
class OutlookQuery(ASTNode):
    """
    OUTLOOK asset=silver horizon=10d [regime=STRESS]

    Produces a trader-facing market outlook using SPIKAN dynamics conditioned
    on the current or requested market backdrop.
    """
    asset: str
    horizon: float  # in years
    regime: Optional[str] = None  # Optional backdrop / regime override
