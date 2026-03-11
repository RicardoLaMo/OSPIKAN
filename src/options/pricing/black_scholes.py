"""
Black-Scholes option pricing model and Greeks.

Pure Python implementation using the math module (no numpy/scipy).
"""

import math
from dataclasses import dataclass
from typing import Literal


@dataclass
class BSResult:
    """Black-Scholes pricing result with Greeks."""
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _standard_normal_cdf(x: float) -> float:
    """
    Compute the cumulative distribution function of standard normal.

    Uses error function (erf) from math module.

    Args:
        x: Input value

    Returns:
        CDF value in [0, 1]
    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _standard_normal_pdf(x: float) -> float:
    """
    Compute the probability density function of standard normal.

    Args:
        x: Input value

    Returns:
        PDF value
    """
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)


def black_scholes(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Black-Scholes option pricing formula.

    Args:
        spot: Current spot price (S)
        strike: Strike price (K)
        time_to_expiry: Time to expiry in years (T)
        volatility: Annualized volatility (sigma)
        risk_free_rate: Risk-free rate (r)
        dividend_yield: Dividend yield (q, default 0)
        option_type: 'call' or 'put'

    Returns:
        Option price
    """
    if time_to_expiry <= 0:
        # Intrinsic value only
        if option_type == "call":
            return max(spot - strike, 0.0)
        else:
            return max(strike - spot, 0.0)

    if volatility <= 0:
        # For zero volatility, use intrinsic value discounted
        if option_type == "call":
            return max(spot * math.exp(-dividend_yield * time_to_expiry) -
                      strike * math.exp(-risk_free_rate * time_to_expiry), 0.0)
        else:
            return max(strike * math.exp(-risk_free_rate * time_to_expiry) -
                      spot * math.exp(-dividend_yield * time_to_expiry), 0.0)

    # Compute d1 and d2
    sqrt_t = math.sqrt(time_to_expiry)
    sqrt_vol = math.sqrt(volatility * volatility)

    d1 = (math.log(spot / strike) +
          (risk_free_rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry) / (sqrt_vol * sqrt_t)
    d2 = d1 - sqrt_vol * sqrt_t

    # Compute option price
    nd1 = _standard_normal_cdf(d1)
    nd2 = _standard_normal_cdf(d2)

    spot_disc = spot * math.exp(-dividend_yield * time_to_expiry)
    strike_disc = strike * math.exp(-risk_free_rate * time_to_expiry)

    if option_type == "call":
        price = spot_disc * nd1 - strike_disc * nd2
    else:
        nd1_neg = _standard_normal_cdf(-d1)
        nd2_neg = _standard_normal_cdf(-d2)
        price = strike_disc * nd2_neg - spot_disc * nd1_neg

    return price


def compute_delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Compute delta (dPrice/dSpot).

    Args:
        spot: Current spot price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        volatility: Annualized volatility
        risk_free_rate: Risk-free rate
        dividend_yield: Dividend yield (default 0)
        option_type: 'call' or 'put'

    Returns:
        Delta value
    """
    if time_to_expiry <= 0:
        # Intrinsic delta
        if option_type == "call":
            return 1.0 if spot > strike else 0.0
        else:
            return -1.0 if spot < strike else 0.0

    if volatility <= 0:
        # Zero volatility case
        if option_type == "call":
            return 1.0 if spot > strike else 0.0
        else:
            return -1.0 if spot < strike else 0.0

    sqrt_t = math.sqrt(time_to_expiry)
    sqrt_vol = math.sqrt(volatility * volatility)

    d1 = (math.log(spot / strike) +
          (risk_free_rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry) / (sqrt_vol * sqrt_t)

    nd1 = _standard_normal_cdf(d1)

    if option_type == "call":
        delta = math.exp(-dividend_yield * time_to_expiry) * nd1
    else:
        nd1_neg = _standard_normal_cdf(-d1)
        delta = -math.exp(-dividend_yield * time_to_expiry) * nd1_neg

    return delta


def compute_gamma(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Compute gamma (d²Price/dSpot²).

    Args:
        spot: Current spot price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        volatility: Annualized volatility
        risk_free_rate: Risk-free rate
        dividend_yield: Dividend yield (default 0)

    Returns:
        Gamma value (same for calls and puts)
    """
    if time_to_expiry <= 0 or volatility <= 0:
        return 0.0

    sqrt_t = math.sqrt(time_to_expiry)
    sqrt_vol = math.sqrt(volatility * volatility)

    d1 = (math.log(spot / strike) +
          (risk_free_rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry) / (sqrt_vol * sqrt_t)

    pdf_d1 = _standard_normal_pdf(d1)
    gamma = math.exp(-dividend_yield * time_to_expiry) * pdf_d1 / (spot * sqrt_vol * sqrt_t)

    return gamma


def compute_vega(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Compute vega (dPrice/dVolatility), per 1% change in volatility.

    Args:
        spot: Current spot price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        volatility: Annualized volatility
        risk_free_rate: Risk-free rate
        dividend_yield: Dividend yield (default 0)

    Returns:
        Vega value per 1% vol change (same for calls and puts)
    """
    if time_to_expiry <= 0 or volatility <= 0:
        return 0.0

    sqrt_t = math.sqrt(time_to_expiry)
    sqrt_vol = math.sqrt(volatility * volatility)

    d1 = (math.log(spot / strike) +
          (risk_free_rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry) / (sqrt_vol * sqrt_t)

    pdf_d1 = _standard_normal_pdf(d1)
    vega = spot * math.exp(-dividend_yield * time_to_expiry) * pdf_d1 * sqrt_t / 100.0

    return vega


def compute_theta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Compute theta (dPrice/dTime), per day (divide by 365).

    Args:
        spot: Current spot price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        volatility: Annualized volatility
        risk_free_rate: Risk-free rate
        dividend_yield: Dividend yield (default 0)
        option_type: 'call' or 'put'

    Returns:
        Theta value per day
    """
    if time_to_expiry <= 0:
        return 0.0

    if volatility <= 0:
        # Intrinsic theta only
        if option_type == "call":
            return risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) / 365.0
        else:
            return -risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) / 365.0

    sqrt_t = math.sqrt(time_to_expiry)
    sqrt_vol = math.sqrt(volatility * volatility)

    d1 = (math.log(spot / strike) +
          (risk_free_rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry) / (sqrt_vol * sqrt_t)
    d2 = d1 - sqrt_vol * sqrt_t

    pdf_d1 = _standard_normal_pdf(d1)
    nd1 = _standard_normal_cdf(d1)
    nd2 = _standard_normal_cdf(d2)
    nd2_neg = _standard_normal_cdf(-d2)

    spot_disc = spot * math.exp(-dividend_yield * time_to_expiry)
    strike_disc = strike * math.exp(-risk_free_rate * time_to_expiry)

    if option_type == "call":
        theta = (-spot_disc * pdf_d1 * sqrt_vol / (2 * sqrt_t) -
                risk_free_rate * strike_disc * nd2 +
                dividend_yield * spot_disc * nd1)
    else:
        nd1_neg = _standard_normal_cdf(-d1)
        theta = (-spot_disc * pdf_d1 * sqrt_vol / (2 * sqrt_t) +
                risk_free_rate * strike_disc * nd2_neg -
                dividend_yield * spot_disc * nd1_neg)

    return theta / 365.0


def compute_rho(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Compute rho (dPrice/dRiskFreeRate), per 1% change in rate.

    Args:
        spot: Current spot price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        volatility: Annualized volatility
        risk_free_rate: Risk-free rate
        dividend_yield: Dividend yield (default 0)
        option_type: 'call' or 'put'

    Returns:
        Rho value per 1% rate change
    """
    if time_to_expiry <= 0 or volatility <= 0:
        # Intrinsic rho
        if option_type == "call":
            return strike * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) / 100.0
        else:
            return -strike * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) / 100.0

    sqrt_t = math.sqrt(time_to_expiry)
    sqrt_vol = math.sqrt(volatility * volatility)

    d1 = (math.log(spot / strike) +
          (risk_free_rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry) / (sqrt_vol * sqrt_t)
    d2 = d1 - sqrt_vol * sqrt_t

    nd2 = _standard_normal_cdf(d2)
    nd2_neg = _standard_normal_cdf(-d2)

    strike_disc = strike * math.exp(-risk_free_rate * time_to_expiry)

    if option_type == "call":
        rho = strike_disc * time_to_expiry * nd2 / 100.0
    else:
        rho = -strike_disc * time_to_expiry * nd2_neg / 100.0

    return rho


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    option_type: Literal["call", "put"] = "call",
) -> BSResult:
    """
    Compute Black-Scholes price and all Greeks.

    Args:
        spot: Current spot price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        volatility: Annualized volatility
        risk_free_rate: Risk-free rate
        dividend_yield: Dividend yield (default 0)
        option_type: 'call' or 'put'

    Returns:
        BSResult with price and Greeks
    """
    price = black_scholes(spot, strike, time_to_expiry, volatility, risk_free_rate, dividend_yield, option_type)
    delta = compute_delta(spot, strike, time_to_expiry, volatility, risk_free_rate, dividend_yield, option_type)
    gamma = compute_gamma(spot, strike, time_to_expiry, volatility, risk_free_rate, dividend_yield)
    vega = compute_vega(spot, strike, time_to_expiry, volatility, risk_free_rate, dividend_yield)
    theta = compute_theta(spot, strike, time_to_expiry, volatility, risk_free_rate, dividend_yield, option_type)
    rho = compute_rho(spot, strike, time_to_expiry, volatility, risk_free_rate, dividend_yield, option_type)

    return BSResult(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
    )
