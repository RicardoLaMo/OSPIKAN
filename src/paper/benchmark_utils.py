from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def git_metadata(repo_root: str | Path) -> dict[str, str | None]:
    root = str(Path(repo_root).resolve())

    def _run(args: list[str]) -> str | None:
        try:
            return (
                subprocess.check_output(args, cwd=root, stderr=subprocess.DEVNULL, text=True)
                .strip()
                .splitlines()[0]
            )
        except Exception:
            return None

    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    ensure_dir(out.parent)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    ensure_dir(out.parent)
    pd.DataFrame(list(rows)).to_csv(out, index=False)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int | None = None) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return 0.0
    if k is None:
        k = int(y_true.sum())
    k = max(1, min(int(k), len(y_true)))
    top_idx = np.argsort(y_score)[-k:]
    return float(y_true[top_idx].mean())


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    rows = calibration_table(y_true=y_true, y_prob=y_prob, n_bins=n_bins)
    total = sum(int(row["count"]) for row in rows)
    if total == 0:
        return 0.0
    return float(
        sum(
            abs(float(row["pred_mean"]) - float(row["obs_rate"])) * int(row["count"]) / total
            for row in rows
        )
    )


def calibration_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    if len(y_true) == 0:
        return []

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []

    for idx in range(n_bins):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        if idx == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        count = int(mask.sum())
        if count == 0:
            continue

        rows.append(
            {
                "bin_left": left,
                "bin_right": right,
                "count": count,
                "pred_mean": float(y_prob[mask].mean()),
                "obs_rate": float(y_true[mask].mean()),
            }
        )

    return rows


@dataclass
class BinaryMetricBundle:
    auc: float
    precision_at_k: float
    brier: float
    ece: float
    n_obs: int
    n_positive: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "auc": self.auc,
            "precision_at_k": self.precision_at_k,
            "brier": self.brier,
            "ece": self.ece,
            "n_obs": self.n_obs,
            "n_positive": self.n_positive,
        }


def summarize_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> BinaryMetricBundle:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    return BinaryMetricBundle(
        auc=safe_auc(y_true, y_prob),
        precision_at_k=precision_at_k(y_true, y_prob),
        brier=brier_score(y_true, y_prob),
        ece=expected_calibration_error(y_true, y_prob),
        n_obs=int(len(y_true)),
        n_positive=int(y_true.sum()),
    )


def make_manifest(
    repo_root: str | Path,
    *,
    benchmark_id: str,
    split_id: str,
    seed_policy: str,
    backend_ids: list[str],
    source_data_refs: list[str],
    owner_doc: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        **git_metadata(repo_root),
        "generated_at_utc": utc_now_iso(),
        "benchmark_id": benchmark_id,
        "split_id": split_id,
        "seed_policy": seed_policy,
        "backend_ids": backend_ids,
        "source_data_refs": source_data_refs,
        "owner_doc": owner_doc,
    }
    if extra:
        payload.update(extra)
    return payload
