"""Update data -> backtest every model -> refit on all data -> forecast -> write outputs/.

    python -m fxviz.pipeline                    # update data, naive + LSTM
    python -m fxviz.pipeline --models naive     # no TensorFlow/Keras needed
    python -m fxviz.pipeline --no-update        # use the cached CSV as it is
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data, outputs
from .evaluation import (
    apply_interval,
    backtest,
    error_quantiles,
    mae_by_horizon,
    metrics,
    skill_score,
)
from .models import LSTMForecaster, NaiveForecaster

log = logging.getLogger("fxviz")


def run_pipeline(
    rates: pd.DataFrame,
    factories: dict,
    horizon: int = config.HORIZON,
    test_size: int = config.TEST_SIZE,
    step: int = config.STEP,
    interval: tuple[float, float] = config.INTERVAL,
    out_dir: Path = config.OUTPUT_DIR,
) -> dict:
    """Run the full evaluation + forecast for every currency pair. ``factories`` maps a
    model name to a zero-argument callable returning a fresh forecaster; the "naive"
    baseline is always included because every skill score is measured against it."""
    factories = {"naive": NaiveForecaster, **factories}
    forecast_rows, metric_rows, horizon_rows = [], [], []

    for pair in config.PAIRS:
        if pair not in rates.columns:
            continue
        series = rates[pair].dropna()
        prices = series.to_numpy(dtype=float)
        future_dates = pd.bdate_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
        per_model = {}

        for name, factory in factories.items():
            started = time.time()
            bt = backtest(factory, prices, horizon, test_size, step)
            m = metrics(bt)
            q_low, q_high = error_quantiles(bt, *interval)

            final_model = factory().fit(prices)
            forecast = final_model.predict(prices, horizon)
            lower, upper = apply_interval(forecast, q_low, q_high)

            per_model[name] = m
            for date, f, lo, hi in zip(future_dates, forecast, lower, upper):
                forecast_rows.append(
                    {"currency": pair, "model": name, "date": date.date().isoformat(),
                     "forecast": f, "lower": lo, "upper": hi}
                )
            for h, err in enumerate(mae_by_horizon(bt), start=1):
                horizon_rows.append({"currency": pair, "model": name, "horizon": h, "mae": err})
            log.info("%s / %s: MAE %.5f (%.1fs)", pair, name, m["mae"], time.time() - started)

        baseline = per_model["naive"]["mae"]
        for name, m in per_model.items():
            metric_rows.append(
                {"currency": pair, "model": name, **m, "skill_vs_naive": skill_score(m["mae"], baseline)}
            )

    info = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_start": rates.index.min().date().isoformat(),
        "data_end": rates.index.max().date().isoformat(),
        "horizon_business_days": horizon,
        "test_size_business_days": test_size,
        "origin_step_business_days": step,
        "interval": list(interval),
        "models": list(factories),
    }
    outputs.save_outputs(
        out_dir,
        pd.DataFrame(forecast_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(horizon_rows),
        info,
    )
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=None, help="comma separated: naive,lstm (default: lstm too if Keras is installed)")
    parser.add_argument("--no-update", action="store_true", help="do not call the API, use data/historical_rates.csv")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    requested = (args.models.split(",") if args.models else ["naive", "lstm"])
    requested = [m.strip() for m in requested if m.strip()]
    factories = {}
    for name in requested:
        if name == "naive":
            continue  # always included
        if name != "lstm":
            parser.error(f"unknown model '{name}'")
        if not LSTMForecaster.available():
            if args.models:
                parser.error("the lstm model needs Keras: pip install -r requirements-train.txt")
            log.warning("Keras not installed: running the naive baseline only.")
            continue
        factories["lstm"] = lambda: LSTMForecaster(
            lookback=config.LOOKBACK, horizon=config.HORIZON, epochs=args.epochs
        )

    if args.no_update:
        rates = data.load_rates()
        if rates.empty:
            parser.error("no cached data; run without --no-update first")
    else:
        log.info("Updating exchange rates...")
        rates = data.update_rates()
    if not data.is_daily(rates):
        parser.error("cached data is weekly (legacy); run without --no-update to download daily data")

    info = run_pipeline(rates, factories)
    log.info("Done. Data up to %s, outputs in %s", info["data_end"], config.OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
