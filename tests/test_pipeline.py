import numpy as np
import pandas as pd

from fxviz import config, outputs
from fxviz.models import Forecaster
from fxviz.pipeline import run_pipeline


class DriftUp(Forecaster):
    name = "drift_up"

    def predict(self, history, horizon):
        return history[-1] * (1 + 0.001 * np.arange(1, horizon + 1))


def test_pipeline_writes_consistent_outputs(tmp_path, synthetic_rates):
    info = run_pipeline(
        synthetic_rates, {"drift_up": DriftUp}, horizon=10, test_size=200, step=5, out_dir=tmp_path
    )
    fc, mt = outputs.load_forecasts(tmp_path), outputs.load_metrics(tmp_path)
    he, run = outputs.load_horizon_errors(tmp_path), outputs.load_run_info(tmp_path)

    assert info["models"] == ["naive", "drift_up"] == run["models"]
    assert set(fc["currency"]) == set(config.PAIRS) and set(fc["model"]) == {"naive", "drift_up"}
    assert len(fc) == len(config.PAIRS) * 2 * 10
    assert (fc["lower"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["upper"]).all()

    # forecasts start the first business day after the data and skip weekends
    first = fc["date"].min()
    assert first == synthetic_rates.index.max() + pd.offsets.BDay(1)
    assert (fc["date"].dt.weekday < 5).all()

    naive = mt[mt["model"] == "naive"]
    assert (naive["skill_vs_naive"] == 0).all()
    assert set(mt.columns) >= {"currency", "model", "mae", "rmse", "mape", "skill_vs_naive", "n_origins"}
    assert len(he) == len(config.PAIRS) * 2 * 10
    assert run["data_end"] == synthetic_rates.index.max().date().isoformat()


def test_naive_forecast_is_flat_at_last_price(tmp_path, synthetic_rates):
    run_pipeline(synthetic_rates, {}, horizon=5, test_size=100, step=5, out_dir=tmp_path)
    fc = outputs.load_forecasts(tmp_path)
    for pair in config.PAIRS:
        flat = fc[(fc["currency"] == pair) & (fc["model"] == "naive")]["forecast"]
        assert np.allclose(flat, synthetic_rates[pair].iloc[-1], rtol=1e-5)
