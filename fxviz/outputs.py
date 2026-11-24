"""Reading and writing the files produced by the pipeline (``outputs/``)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config

FORECASTS = "forecasts.csv"
METRICS = "metrics.csv"
HORIZON_ERRORS = "horizon_errors.csv"
RUN_INFO = "run_info.json"


def save_outputs(out_dir: Path, forecasts: pd.DataFrame, metrics: pd.DataFrame, horizon_errors: pd.DataFrame, info: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(out_dir / FORECASTS, index=False, float_format="%.6f")
    metrics.to_csv(out_dir / METRICS, index=False, float_format="%.6f")
    horizon_errors.to_csv(out_dir / HORIZON_ERRORS, index=False, float_format="%.6f")
    (out_dir / RUN_INFO).write_text(json.dumps(info, indent=2), encoding="utf-8")


def _read_csv(name: str, out_dir: Path, **kwargs):
    path = Path(out_dir) / name
    return pd.read_csv(path, **kwargs) if path.exists() else None


def load_forecasts(out_dir: Path = config.OUTPUT_DIR):
    return _read_csv(FORECASTS, out_dir, parse_dates=["date"])


def load_metrics(out_dir: Path = config.OUTPUT_DIR):
    return _read_csv(METRICS, out_dir)


def load_horizon_errors(out_dir: Path = config.OUTPUT_DIR):
    return _read_csv(HORIZON_ERRORS, out_dir)


def load_run_info(out_dir: Path = config.OUTPUT_DIR):
    path = Path(out_dir) / RUN_INFO
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
