#!/usr/bin/env python3
"""
Run a paper-oriented quantitative benchmark on the existing silver/SPIKAN stack.

This is a thin evidence harness, not a replacement for the current training
script. It reuses the current SPIKAN data prep and model training code, adds
tabular comparison baselines, and exports benchmark artifacts under
`output/paper/quant/`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paper.benchmark_utils import (  # noqa: E402
    calibration_table,
    compact_utc_timestamp,
    ensure_dir,
    make_manifest,
    summarize_binary_metrics,
    write_csv,
    write_json,
)


def _load_train_spikan_module():
    module_path = REPO_ROOT / "scripts" / "silver" / "train_spikan.py"
    spec = importlib.util.spec_from_file_location("paper_train_spikan", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load SPIKAN training module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _split_arrays(data: dict[str, Any], split: str) -> dict[str, Any]:
    assets, macro, geom, time, reg_target, shock = data[split]
    features = torch.cat([assets, macro, geom, time], dim=1).numpy()
    return {
        "X": features,
        "reg_target": reg_target.squeeze(-1).numpy(),
        "shock_raw": shock.numpy().astype(int),
        "shock_binary": (shock.numpy() > 0).astype(int),
        "shock_severe": (shock.numpy() > 1).astype(int),
        "dates": pd.Index(data["dates"][split]),
    }


def _calibrate_scores(val_scores: np.ndarray, val_labels: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    calibrator = LogisticRegression(random_state=42, max_iter=1000)
    calibrator.fit(np.asarray(val_scores, dtype=float).reshape(-1, 1), np.asarray(val_labels).astype(int))
    return calibrator.predict_proba(np.asarray(test_scores, dtype=float).reshape(-1, 1))[:, 1]


def _predict_spikan_scores(model, split_tensors, device: torch.device, batch_size: int) -> np.ndarray:
    dataset = TensorDataset(*split_tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            assets, macro, geom, time, _reg_target, _shock = [b.to(device) for b in batch]
            pred = model(assets, macro, geom, time).detach().cpu().numpy().reshape(-1)
            outputs.append(np.abs(pred))
    return np.concatenate(outputs) if outputs else np.array([], dtype=float)


def _clone_with_zero_geometry(data: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {"dates": data["dates"]}
    for split in ("train", "val", "test"):
        assets, macro, geom, time, reg_target, shock = data[split]
        cloned[split] = (
            assets.clone(),
            macro.clone(),
            torch.zeros_like(geom),
            time.clone(),
            reg_target.clone(),
            shock.clone(),
        )
    return cloned


def _fit_tabular_baselines(train: dict[str, Any]) -> dict[str, Any]:
    x_train = train["X"]
    y_train = train["shock_binary"]

    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=2000)),
        ]
    )
    logistic.fit(x_train, y_train)

    forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    forest.fit(x_train, y_train)

    return {"logistic_regression": logistic, "random_forest": forest}


def _stress_rows(backend: str, severe_true: np.ndarray, severe_prob: np.ndarray) -> list[dict[str, Any]]:
    summary = summarize_binary_metrics(severe_true, severe_prob).as_dict()
    return [
        {
            "backend": backend,
            "slice": "severe_shock_vs_rest",
            **summary,
        }
    ]


def run_benchmark(args: argparse.Namespace) -> Path:
    module = _load_train_spikan_module()

    features_path = Path(args.features) if args.features else None
    if features_path is None:
        features = module.load_features(args.config)
        source_data_refs = [str(Path(args.config).resolve())]
    else:
        features = pd.read_parquet(features_path)
        source_data_refs = [str(features_path.resolve())]

    training_config = module.TrainingConfig(
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=str((REPO_ROOT / "reports" / "silver" / "paper_quant_spikan").resolve()),
        use_amp=(not args.cpu),
        multi_gpu=False,
        num_workers=0,
        compile_model=(not args.no_compile),
        patience=args.patience,
    )

    data = module.prepare_training_data(features, training_config)
    split_train = _split_arrays(data, "train")
    split_val = _split_arrays(data, "val")
    split_test = _split_arrays(data, "test")

    benchmark_id = f"quant_{compact_utc_timestamp()}"
    output_root = ensure_dir(REPO_ROOT / args.output_root)
    artifact_root = ensure_dir(output_root)

    if args.cpu:
        device = torch.device("cpu")
        num_gpus = 0
    else:
        device, num_gpus = module.setup_cuda_environment()

    leaderboard_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    casebook_rows: list[dict[str, Any]] = []
    backend_ids: list[str] = []

    spikan_test_prob = None
    spikan_metrics = None

    if not args.skip_spikan:
        backend_ids.append("spikan")
        model, _history = module.train_model(data, training_config, device, num_gpus=num_gpus)
        val_scores = _predict_spikan_scores(model, data["val"], device, args.batch_size)
        test_scores = _predict_spikan_scores(model, data["test"], device, args.batch_size)
        spikan_test_prob = _calibrate_scores(val_scores, split_val["shock_binary"], test_scores)
        spikan_metrics = summarize_binary_metrics(split_test["shock_binary"], spikan_test_prob)
        leaderboard_rows.append(
            {
                "backend": "spikan",
                "task": "silver_shock_prediction",
                "split": "test",
                "primary_metric": "auc",
                "primary_metric_value": spikan_metrics.auc,
                **spikan_metrics.as_dict(),
            }
        )
        for row in calibration_table(split_test["shock_binary"], spikan_test_prob):
            calibration_rows.append({"backend": "spikan", **row})
        stress_rows.extend(_stress_rows("spikan", split_test["shock_severe"], spikan_test_prob))

        top_idx = np.argsort(spikan_test_prob)[- min(args.casebook_size, len(spikan_test_prob)) :]
        for idx in top_idx[::-1]:
            casebook_rows.append(
                {
                    "backend": "spikan",
                    "date": str(split_test["dates"][idx]),
                    "shock_label": int(split_test["shock_raw"][idx]),
                    "shock_probability": float(spikan_test_prob[idx]),
                }
            )

        if args.include_ablation:
            zero_geom_data = _clone_with_zero_geometry(data)
            ablation_model, _ = module.train_model(zero_geom_data, training_config, device, num_gpus=num_gpus)
            ablation_val_scores = _predict_spikan_scores(ablation_model, zero_geom_data["val"], device, args.batch_size)
            ablation_test_scores = _predict_spikan_scores(ablation_model, zero_geom_data["test"], device, args.batch_size)
            ablation_prob = _calibrate_scores(ablation_val_scores, split_val["shock_binary"], ablation_test_scores)
            ablation_metrics = summarize_binary_metrics(split_test["shock_binary"], ablation_prob)
            ablation_rows.append(
                {
                    "backend": "spikan",
                    "ablation": "no_geometry",
                    "metric": "auc",
                    "full_model_value": spikan_metrics.auc,
                    "ablated_value": ablation_metrics.auc,
                    "delta": ablation_metrics.auc - spikan_metrics.auc,
                }
            )
            ablation_rows.append(
                {
                    "backend": "spikan",
                    "ablation": "no_geometry",
                    "metric": "precision_at_k",
                    "full_model_value": spikan_metrics.precision_at_k,
                    "ablated_value": ablation_metrics.precision_at_k,
                    "delta": ablation_metrics.precision_at_k - spikan_metrics.precision_at_k,
                }
            )

    baselines = _fit_tabular_baselines(split_train)
    for backend, model in baselines.items():
        backend_ids.append(backend)
        prob = model.predict_proba(split_test["X"])[:, 1]
        metrics = summarize_binary_metrics(split_test["shock_binary"], prob)
        leaderboard_rows.append(
            {
                "backend": backend,
                "task": "silver_shock_prediction",
                "split": "test",
                "primary_metric": "auc",
                "primary_metric_value": metrics.auc,
                **metrics.as_dict(),
            }
        )
        for row in calibration_table(split_test["shock_binary"], prob):
            calibration_rows.append({"backend": backend, **row})
        stress_rows.extend(_stress_rows(backend, split_test["shock_severe"], prob))

    write_csv(artifact_root / "leaderboard.csv", leaderboard_rows)
    write_csv(artifact_root / "calibration.csv", calibration_rows)
    write_csv(artifact_root / "stress_slices.csv", stress_rows)
    write_csv(artifact_root / "ablations.csv", ablation_rows)
    write_csv(artifact_root / "casebook.csv", casebook_rows)

    manifest = make_manifest(
        REPO_ROOT,
        benchmark_id=benchmark_id,
        split_id=f"{args.train_end}|{args.val_start}:{args.val_end}|{args.test_start}",
        seed_policy="fixed:42",
        backend_ids=backend_ids,
        source_data_refs=source_data_refs,
        owner_doc="docs/paper/QUANT_BENCHMARK_SPEC.md",
        extra={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "include_ablation": bool(args.include_ablation),
            "n_train": int(len(split_train["X"])),
            "n_val": int(len(split_val["X"])),
            "n_test": int(len(split_test["X"])),
            "features_path": str(features_path.resolve()) if features_path else None,
        },
    )
    write_json(artifact_root / "manifest.json", manifest)

    summary = {
        "artifact_root": str(artifact_root),
        "benchmark_id": benchmark_id,
        "leaderboard": leaderboard_rows,
        "spikan_available": spikan_metrics is not None,
    }
    print(json.dumps(summary, indent=2))
    return artifact_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper quantitative benchmark for SPIKAN and baselines.")
    parser.add_argument("--config", default="configs/silver_universe.yaml", help="Universe config used to locate features.")
    parser.add_argument("--features", default=None, help="Parquet feature table to use directly.")
    parser.add_argument("--output-root", default="output/paper/quant", help="Artifact root for benchmark outputs.")
    parser.add_argument("--train-end", default="2019-12-31")
    parser.add_argument("--val-start", default="2020-01-01")
    parser.add_argument("--val-end", default="2021-12-31")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--casebook-size", type=int, default=25)
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile in SPIKAN training.")
    parser.add_argument("--skip-spikan", action="store_true", help="Only run tabular baselines.")
    parser.add_argument("--include-ablation", action="store_true", help="Train a no-geometry SPIKAN ablation.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
