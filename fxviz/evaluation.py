"""Out-of-sample evaluation of forecasters.

Protocol (rolling origin, no look-ahead):
  1. the model is fitted on everything *before* the test window;
  2. every ``step`` business days inside the test window we forecast ``horizon`` days
     ahead using only data available at that origin, and compare with what really happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .models import Forecaster


@dataclass
class BacktestResult:
    origins: np.ndarray  # index of the first forecasted day for each origin
    actual: np.ndarray  # (n_origins, horizon)
    pred: np.ndarray  # (n_origins, horizon)

    @property
    def log_errors(self) -> np.ndarray:
        return np.log(self.actual / self.pred)


def backtest(
    factory: Callable[[], Forecaster],
    prices: np.ndarray,
    horizon: int,
    test_size: int,
    step: int,
) -> BacktestResult:
    prices = np.asarray(prices, dtype=float)
    if test_size < horizon:
        raise ValueError("test_size must be at least the forecast horizon.")
    split = len(prices) - test_size
    if split < 2 * horizon:
        raise ValueError("Not enough history left for training after reserving the test window.")

    model = factory().fit(prices[:split])
    origins = np.arange(split, len(prices) - horizon + 1, step)
    pred = np.stack([model.predict(prices[:o], horizon) for o in origins])
    actual = np.stack([prices[o : o + horizon] for o in origins])
    return BacktestResult(origins=origins, actual=actual, pred=pred)


def metrics(result: BacktestResult) -> dict[str, float]:
    err = result.pred - result.actual
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mape": float((np.abs(err) / result.actual).mean() * 100),
        "n_origins": int(len(result.origins)),
    }


def mae_by_horizon(result: BacktestResult) -> np.ndarray:
    return np.abs(result.pred - result.actual).mean(axis=0)


def skill_score(mae: float, baseline_mae: float) -> float:
    """1 - MAE/MAE_baseline. Positive: better than the baseline. Zero: same. Negative: worse."""
    if baseline_mae == 0:
        return 0.0 if mae == 0 else float("-inf")
    return 1.0 - mae / baseline_mae


def error_quantiles(result: BacktestResult, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-horizon quantiles of log(actual/forecast), used to build prediction intervals."""
    q = np.quantile(result.log_errors, [lower, upper], axis=0)
    return q[0], q[1]


def apply_interval(forecast: np.ndarray, q_low: np.ndarray, q_high: np.ndarray):
    return forecast * np.exp(q_low), forecast * np.exp(q_high)
