import numpy as np
import pytest

from fxviz.models import LSTMForecaster, NaiveForecaster


def test_naive_repeats_last_value():
    out = NaiveForecaster().fit(np.array([1.0, 2.0])).predict(np.array([1.0, 1.1, 1.2]), 4)
    assert np.allclose(out, 1.2) and out.shape == (4,)


class StubKeras:
    """Stands in for a Keras model so the data handling can be tested without TensorFlow."""

    def __init__(self):
        self.fit_args = None

    def fit(self, X, y, **kwargs):
        self.fit_args = (X, y, kwargs)

    def predict(self, x, verbose=0):
        return np.full((len(x), 5), 0.5)


@pytest.fixture
def lstm(monkeypatch):
    model = LSTMForecaster(lookback=10, horizon=5, epochs=1)
    stub = StubKeras()
    monkeypatch.setattr(model, "_build_model", lambda: stub)
    monkeypatch.setattr(model, "_callbacks", lambda: ["cb"])
    model.stub = stub
    return model


def test_windows_targets_are_the_future_cumulative_log_returns():
    model = LSTMForecaster(lookback=4, horizon=3)
    prices = np.exp(np.cumsum(np.arange(1, 21) / 100))  # known returns 0.02 ... 0.20
    X, y, scale = model.make_windows(prices)
    returns = np.diff(np.log(prices))
    assert X.shape == (len(returns) - 4 - 3 + 1, 4, 1) and y.shape == (X.shape[0], 3)
    i = 2
    assert np.allclose(X[i, :, 0] * scale, returns[i : i + 4])
    assert np.allclose(y[i] * scale, np.cumsum(returns[i + 4 : i + 7]))  # strictly after the input window


def test_windows_need_enough_history():
    with pytest.raises(ValueError):
        LSTMForecaster(lookback=60, horizon=30).make_windows(np.ones(50))


def test_fit_uses_time_ordered_validation_with_gap(lstm):
    rng = np.random.default_rng(1)
    prices = np.exp(np.cumsum(rng.normal(0, 0.005, 800)))
    lstm.fit(prices)
    X, y, kwargs = lstm.stub.fit_args
    Xv, yv = kwargs["validation_data"]
    n_total = len(prices) - 1 - 10 - 5 + 1
    n_val = int(n_total * 0.15)
    assert len(Xv) == n_val
    assert len(X) == n_total - n_val - (10 + 5)  # gap = lookback + horizon, no overlap with validation
    assert kwargs["callbacks"] == ["cb"]


def test_predict_converts_cumulative_returns_back_to_prices(lstm):
    rng = np.random.default_rng(2)
    prices = np.exp(np.cumsum(rng.normal(0, 0.005, 300)))
    lstm.fit(prices)
    out = lstm.predict(prices, 5)
    assert out.shape == (5,)
    assert np.allclose(out, prices[-1] * np.exp(0.5 * lstm.scale_))  # stub returns 0.5 (scaled units)
    assert lstm.predict(prices, 3).shape == (3,)


def test_predict_guards(lstm):
    with pytest.raises(RuntimeError):
        lstm.predict(np.ones(50), 5)
    lstm.fit(np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.005, 300))))
    with pytest.raises(ValueError):
        lstm.predict(np.ones(50), 99)  # longer than trained horizon
    with pytest.raises(ValueError):
        lstm.predict(np.ones(5), 5)  # too little history
