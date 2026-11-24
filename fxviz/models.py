"""Forecasting models. Every model works on a 1-D array of prices (oldest first)."""

from __future__ import annotations

import importlib.util

import numpy as np


class Forecaster:
    """Minimal interface: ``fit`` on training prices, ``predict`` from any history."""

    name = "base"

    def fit(self, prices: np.ndarray) -> "Forecaster":
        return self

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        raise NotImplementedError


class NaiveForecaster(Forecaster):
    """Random-walk baseline: tomorrow equals today. Hard to beat for exchange rates."""

    name = "naive"

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        return np.full(horizon, float(history[-1]))


class LSTMForecaster(Forecaster):
    """LSTM that maps the last ``lookback`` daily log-returns to the next ``horizon``
    cumulative log-returns (direct multi-horizon, no recursive error accumulation).

    Working on returns instead of price levels keeps the inputs stationary and lets the
    model forecast rates outside the range seen in training (a MinMax-scaled level
    model cannot). Keras is imported lazily, so the rest of the project does not need it.
    """

    name = "lstm"

    def __init__(
        self,
        lookback: int = 60,
        horizon: int = 30,
        units: int = 32,
        epochs: int = 60,
        batch_size: int = 64,
        patience: int = 8,
        val_fraction: float = 0.15,
        seed: int = 42,
        verbose: int = 0,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed
        self.verbose = verbose
        self.model_ = None
        self.scale_: float | None = None

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("keras") is not None

    # -- data preparation (pure numpy, unit-tested) ---------------------------------
    def make_windows(self, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        returns = np.diff(np.log(np.asarray(prices, dtype=float)))
        n = len(returns) - self.lookback - self.horizon + 1
        if n <= 0:
            raise ValueError(
                f"Need more than {self.lookback + self.horizon + 1} prices, got {len(prices)}."
            )
        scale = float(returns.std()) or 1.0
        X = np.stack([returns[i : i + self.lookback] for i in range(n)])
        y = np.stack(
            [np.cumsum(returns[i + self.lookback : i + self.lookback + self.horizon]) for i in range(n)]
        )
        return (X / scale)[..., None], y / scale, scale

    # -- keras bits (kept tiny and isolated) ----------------------------------------
    def _build_model(self):
        import keras
        from keras import layers

        keras.utils.set_random_seed(self.seed)
        model = keras.Sequential(
            [
                layers.Input(shape=(self.lookback, 1)),
                layers.LSTM(self.units),
                layers.Dropout(0.2),
                layers.Dense(self.horizon),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        return model

    def _callbacks(self):
        import keras

        return [
            keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=max(2, self.patience // 2)),
        ]

    # -- public API --------------------------------------------------------------------
    def fit(self, prices: np.ndarray) -> "LSTMForecaster":
        X, y, self.scale_ = self.make_windows(prices)
        n = len(X)
        n_val = int(n * self.val_fraction)
        # Targets span `horizon` days and inputs span `lookback` days: leave a gap so that no
        # validation sample overlaps the training samples (time-ordered split, no shuffling).
        gap = self.lookback + self.horizon
        n_train = n - n_val - gap
        self.model_ = self._build_model()
        if n_val >= 20 and n_train >= 100:
            validation = (X[n - n_val :], y[n - n_val :])
            X_train, y_train = X[:n_train], y[:n_train]
            callbacks = self._callbacks()
        else:
            validation, X_train, y_train, callbacks = None, X, y, []
        self.model_.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation,
            callbacks=callbacks,
            verbose=self.verbose,
        )
        return self

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        if horizon > self.horizon:
            raise ValueError(f"Model was trained for horizon {self.horizon}, asked for {horizon}.")
        history = np.asarray(history, dtype=float)
        if len(history) < self.lookback + 1:
            raise ValueError(f"Need at least {self.lookback + 1} prices of history.")
        returns = np.diff(np.log(history[-(self.lookback + 1) :]))
        x = (returns / self.scale_).reshape(1, self.lookback, 1)
        cumulative = np.asarray(self.model_.predict(x, verbose=0), dtype=float)[0] * self.scale_
        return history[-1] * np.exp(cumulative[:horizon])
