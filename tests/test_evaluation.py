import numpy as np
import pytest

from fxviz import evaluation as ev
from fxviz.models import Forecaster, NaiveForecaster


class Spy(Forecaster):
    """Records what the backtest feeds the model, to prove there is no look-ahead."""

    name = "spy"
    fit_len = None
    history_lens = []

    def fit(self, prices):
        Spy.fit_len = len(prices)
        Spy.history_lens = []
        return self

    def predict(self, history, horizon):
        Spy.history_lens.append(len(history))
        return np.full(horizon, history[-1])


def test_backtest_has_no_lookahead():
    prices = np.linspace(1, 2, 300)
    res = ev.backtest(Spy, prices, horizon=10, test_size=100, step=5)
    assert Spy.fit_len == 200  # trained strictly before the test window
    assert np.array_equal(res.origins, np.arange(200, 291, 5))
    assert Spy.history_lens == list(res.origins)  # at origin o the model sees prices[:o] only
    assert np.array_equal(res.actual[0], prices[200:210])
    assert np.array_equal(res.actual[-1], prices[290:300])  # last window ends exactly at the data end


def test_naive_on_constant_series_is_perfect():
    res = ev.backtest(NaiveForecaster, np.full(300, 1.1), 10, 100, 5)
    m = ev.metrics(res)
    assert m["mae"] == m["rmse"] == m["mape"] == 0


def test_metrics_values():
    res = ev.BacktestResult(
        origins=np.array([0]),
        actual=np.array([[2.0, 2.0]]),
        pred=np.array([[1.0, 3.0]]),
    )
    m = ev.metrics(res)
    assert m["mae"] == 1.0 and m["rmse"] == 1.0 and m["mape"] == 50.0 and m["n_origins"] == 1


def test_mae_grows_with_horizon_for_a_random_walk():
    rng = np.random.default_rng(0)
    prices = np.exp(np.cumsum(rng.normal(0, 0.005, 3000)))
    res = ev.backtest(NaiveForecaster, prices, 30, 2000, 1)
    by_h = ev.mae_by_horizon(res)
    assert by_h[-1] > 3 * by_h[0]  # roughly sqrt(30) ~ 5.5


def test_skill_score():
    assert ev.skill_score(0.5, 1.0) == 0.5
    assert ev.skill_score(1.0, 1.0) == 0.0
    assert ev.skill_score(2.0, 1.0) == -1.0
    assert ev.skill_score(0.0, 0.0) == 0.0


def test_interval_covers_roughly_the_nominal_share():
    rng = np.random.default_rng(1)
    prices = np.exp(np.cumsum(rng.normal(0, 0.005, 5000)))
    train = ev.backtest(NaiveForecaster, prices[:3000], 20, 2500, 1)
    lo_q, hi_q = ev.error_quantiles(train, 0.1, 0.9)
    assert (lo_q < 0).all() and (hi_q > 0).all()
    assert (np.diff(hi_q) > 0).mean() > 0.9  # band widens with the horizon

    fresh = ev.backtest(NaiveForecaster, prices[3000:], 20, 1900, 1)
    lower, upper = ev.apply_interval(fresh.pred, lo_q, hi_q)
    coverage = ((fresh.actual >= lower) & (fresh.actual <= upper)).mean()
    assert 0.72 < coverage < 0.88


def test_backtest_validates_sizes():
    with pytest.raises(ValueError):
        ev.backtest(NaiveForecaster, np.ones(100), horizon=30, test_size=10, step=1)
    with pytest.raises(ValueError):
        ev.backtest(NaiveForecaster, np.ones(100), horizon=30, test_size=90, step=1)
