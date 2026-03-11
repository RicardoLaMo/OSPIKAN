from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _build_color_map(labels) -> Dict[object, tuple]:
    uniq = list(pd.unique(pd.Series(labels).dropna()))
    cmap = plt.get_cmap("tab10" if len(uniq) <= 10 else "tab20")
    return {lab: cmap(i % cmap.N) for i, lab in enumerate(uniq)}


def plot_price_with_regimes(
    price: pd.Series,
    regimes: pd.Series,
    *,
    title: str,
    out_path: str,
    ylabel: str = "Price",
    color_map: Optional[Dict[object, tuple]] = None,
) -> None:
    df = pd.concat([price.rename("price"), regimes.rename("regime")], axis=1).dropna()
    if df.empty:
        raise ValueError("No overlapping non-null data to plot.")

    cm = color_map or _build_color_map(df["regime"])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["price"], color="black", linewidth=1.2)

    # Shade contiguous regime segments.
    r = df["regime"]
    start = df.index[0]
    current = r.iloc[0]
    for i in range(1, len(df.index)):
        if r.iloc[i] != current:
            ax.axvspan(start, df.index[i], color=cm.get(current, (0.8, 0.8, 0.8, 1.0)), alpha=0.18)
            start = df.index[i]
            current = r.iloc[i]
    ax.axvspan(start, df.index[-1], color=cm.get(current, (0.8, 0.8, 0.8, 1.0)), alpha=0.18)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_timeseries(
    series: pd.Series,
    *,
    title: str,
    out_path: str,
    ylabel: str,
) -> None:
    s = series.dropna()
    if s.empty:
        raise ValueError("Series is empty after dropping NaNs.")

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(s.index, s.values, linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_regime_probabilities(
    probs: pd.DataFrame,
    *,
    title: str,
    out_path: str,
) -> None:
    """
    Plots regime probabilities over time (one line per regime).
    """
    p = probs.dropna()
    if p.empty:
        raise ValueError("Probability table is empty after dropping NaNs.")

    fig, ax = plt.subplots(figsize=(12, 4))
    for col in p.columns:
        ax.plot(p.index, p[col], linewidth=1.1, label=str(col))

    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncols=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# Fluid dynamics visualization helpers

FLUID_COLORS = {
    "STABLE": "#2ecc71",
    "TRANSITION": "#f39c12",
    "STRESS": "#e74c3c",
    "RECOVERY": "#3498db",
    "SHOCK_IMMINENT": "#9b59b6",
    "MOMENTUM_PERSISTING": "#1abc9c",
    "MOMENTUM_DECAYING": "#e67e22",
    "NORMAL": "#95a5a6",
    "UNKNOWN": "#bdc3c7",
}


def plot_shock_formation_index(
    sfi: pd.Series,
    price: Optional[pd.Series] = None,
    *,
    threshold: float = 2.0,
    title: str = "Shock Formation Index",
    out_path: str,
) -> None:
    """
    Plots shock formation index with threshold highlighting.

    Args:
        sfi: Shock formation index (z-scored)
        price: Optional price series for overlay
        threshold: Z-score threshold for shock warnings
        title: Plot title
        out_path: Output file path
    """
    sfi = sfi.dropna()
    if sfi.empty:
        raise ValueError("SFI series is empty after dropping NaNs.")

    fig, ax1 = plt.subplots(figsize=(12, 4))

    # Plot SFI
    ax1.fill_between(sfi.index, sfi.values, threshold,
                     where=sfi.values > threshold,
                     alpha=0.5, color=FLUID_COLORS["SHOCK_IMMINENT"],
                     label="Shock Warning")
    ax1.plot(sfi.index, sfi.values, color="purple", linewidth=1.0, alpha=0.8)
    ax1.axhline(y=threshold, color="r", linestyle="--", alpha=0.5, label=f"Threshold (z={threshold})")
    ax1.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    ax1.set_ylabel("SFI (z-score)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.25)

    # Overlay price if provided
    if price is not None:
        price = price.loc[sfi.index].dropna()
        ax2 = ax1.twinx()
        ax2.plot(price.index, price.values, "k-", linewidth=1.2, alpha=0.7, label="Price")
        ax2.set_ylabel("Price")
        ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_fluid_dynamics_panel(
    features: pd.DataFrame,
    *,
    title: str = "Fluid Dynamics Features",
    out_path: str,
) -> None:
    """
    Multi-panel plot of key fluid dynamics features.

    Args:
        features: DataFrame with fluid dynamics columns
        title: Overall title
        out_path: Output file path
    """
    # Determine which features are available
    panels = []
    if "shock_formation_index_z" in features.columns:
        panels.append(("shock_formation_index_z", "SFI (z-score)", "purple"))
    if "momentum_decay_rate" in features.columns:
        panels.append(("momentum_decay_rate", "Decay Rate", "orange"))
    if "effective_viscosity" in features.columns:
        panels.append(("effective_viscosity", "Viscosity", "teal"))
    if "flow_divergence" in features.columns:
        panels.append(("flow_divergence", "Flow Divergence", "green"))

    if not panels:
        raise ValueError("No fluid dynamics features found in DataFrame.")

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (col, label, color) in zip(axes, panels):
        data = features[col].dropna()
        ax.plot(data.index, data.values, color=color, linewidth=0.9, alpha=0.8)

        if col == "flow_divergence":
            ax.fill_between(data.index, data.values, 0,
                           where=data.values > 0, alpha=0.3, color="green")
            ax.fill_between(data.index, data.values, 0,
                           where=data.values < 0, alpha=0.3, color="red")
        elif col == "shock_formation_index_z":
            ax.fill_between(data.index, data.values, 2,
                           where=data.values > 2, alpha=0.5, color=FLUID_COLORS["SHOCK_IMMINENT"])
            ax.axhline(y=2, color="r", linestyle="--", alpha=0.5)

        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Date")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_static_vs_dynamic(
    ricci: pd.Series,
    sfi: pd.Series,
    forward_returns: Optional[pd.Series] = None,
    *,
    title: str = "Static (Ricci) vs Dynamic (SFI) Features",
    out_path: str,
) -> None:
    """
    Scatter plot comparing Ricci curvature and SFI.

    Args:
        ricci: Ricci curvature series
        sfi: Shock formation index (z-scored)
        forward_returns: Optional forward returns for coloring
        title: Plot title
        out_path: Output file path
    """
    common_idx = ricci.dropna().index.intersection(sfi.dropna().index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping data between Ricci and SFI.")

    ricci_c = ricci.loc[common_idx]
    sfi_c = sfi.loc[common_idx]

    fig, ax = plt.subplots(figsize=(10, 8))

    if forward_returns is not None:
        fwd = forward_returns.loc[common_idx].dropna()
        common_idx2 = ricci_c.index.intersection(fwd.index)
        scatter = ax.scatter(ricci_c.loc[common_idx2], sfi_c.loc[common_idx2],
                            c=fwd.loc[common_idx2] * 100, cmap="RdYlGn",
                            alpha=0.5, s=10, vmin=-5, vmax=5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Forward Return (%)")
    else:
        ax.scatter(ricci_c, sfi_c, alpha=0.3, s=10, c="blue")

    ax.axhline(y=2, color="r", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Ricci Curvature (Static Geometry)")
    ax.set_ylabel("SFI Z-Score (Dynamic Physics)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
