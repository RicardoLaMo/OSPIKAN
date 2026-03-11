#!/bin/sh
set -eu

# Resolve repo root regardless of current working directory.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Avoid noisy Matplotlib temp-cache warnings when the default cache dir isn't writable.
MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig"
export MPLCONFIGDIR
mkdir -p "${MPLCONFIGDIR}"

# End-to-end runner for the silver trend/regime pipeline.
#
# Requires network access for the `ingest` step.
# Usage:
#   scripts/run_silver_end_to_end.sh [config_path] [reports_root]
#
# Example:
#   scripts/run_silver_end_to_end.sh configs/silver_universe.yaml reports/silver/runs

CONFIG_PATH="${1:-configs/silver_universe.yaml}"
REPORTS_ROOT="${2:-reports/silver/runs}"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  echo "Run from repo root or pass an absolute path, e.g.:" >&2
  echo "  sh scripts/run_silver_end_to_end.sh \"${REPO_ROOT}/configs/silver_universe.yaml\"" >&2
  exit 1
fi

echo "Config: ${CONFIG_PATH}"
echo "Reports root: ${REPORTS_ROOT}"

if [ "${PANEL_PATH:-}" != "" ]; then
  if [ ! -f "${PANEL_PATH}" ]; then
    echo "PANEL_PATH not found: ${PANEL_PATH}" >&2
    exit 1
  fi
  echo "Using existing panel: ${PANEL_PATH}"
else
  # Ingest requires at least one data backend.
  python - <<'PY'
import importlib
import sys

def has(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False

if not (has("openbb") or has("yfinance")):
    print("ERROR: ingest requires `openbb` and/or `yfinance`, but neither is importable in this Python env.", file=sys.stderr)
    print("Fix:", file=sys.stderr)
    print("  - Install deps: pip install -r requirements.txt", file=sys.stderr)
    print("  - Or minimal ingest deps: pip install yfinance", file=sys.stderr)
    print("Workaround (use existing data):", file=sys.stderr)
    print("  - Run with PANEL_PATH=... to skip ingest/align", file=sys.stderr)
    sys.exit(1)
PY

  python src/pipeline/silver_pipeline.py ingest --config "${CONFIG_PATH}" >/dev/null
  PANEL_PATH="$(python src/pipeline/silver_pipeline.py align --config "${CONFIG_PATH}")"
fi

RUN_ID="$(basename "${PANEL_PATH}")"
RUN_ID="${RUN_ID#silver_panel_close_}"
RUN_ID="${RUN_ID%.csv}"

OUT_DIR="${REPORTS_ROOT}/${RUN_ID}"

echo "Run id: ${RUN_ID}"
echo "Panel: ${PANEL_PATH}"
echo "Out dir: ${OUT_DIR}"

python src/pipeline/silver_pipeline.py qa \
  --panel "${PANEL_PATH}" \
  --required "SI=F,GC=F,DX-Y.NYB,^TNX,SPY"

if [ "${FEATURES_PATH:-}" != "" ]; then
  if [ ! -f "${FEATURES_PATH}" ]; then
    echo "FEATURES_PATH not found: ${FEATURES_PATH}" >&2
    exit 1
  fi
  echo "Using existing features: ${FEATURES_PATH}"
else
  FEATURES_PATH="$(python src/pipeline/silver_pipeline.py features --panel "${PANEL_PATH}")"
  echo "Features: ${FEATURES_PATH}"
fi

python scripts/silver_report.py --features "${FEATURES_PATH}" --out-dir "${OUT_DIR}"
python scripts/silver_markov_regimes.py --features "${FEATURES_PATH}" --k 3 --out-dir "${OUT_DIR}"
python scripts/silver_geometric_regimes.py --features "${FEATURES_PATH}" --out-dir "${OUT_DIR}"
python scripts/analyze_geometric_signals.py --features "${FEATURES_PATH}" --out-dir "${OUT_DIR}/signals"
python scripts/compare_regime_methods.py --features "${FEATURES_PATH}" --out-dir "${OUT_DIR}/analysis"
python scripts/ricci_flow_shock_scenarios.py --panel "${PANEL_PATH}" --out-dir "${OUT_DIR}/ricci_flow"

echo "Done."
echo "Reports: ${OUT_DIR}"
