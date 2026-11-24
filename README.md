# 💱 Exchange Rate Visualizer

[![CI](https://github.com/JM181998/exchange-rate-visualizer/actions/workflows/ci.yml/badge.svg)](https://github.com/JM181998/exchange-rate-visualizer/actions/workflows/ci.yml)

Interactive Streamlit dashboard to explore **EUR exchange rates against USD, AUD, GBP and CHF since 1999**, plus a 30-business-day forecast that is **honestly benchmarked** against a "no change" baseline.

- 📈 Daily ECB reference rates (via the [Frankfurter API](https://frankfurter.dev)), updated incrementally.
- 🔍 Rate or rebased-to-100 view, period presets, headline metrics, CSV export.
- 🔮 Forecasts from a random-walk baseline and an LSTM, with 80 % prediction intervals.
- 📊 Rolling-origin backtest: MAE, RMSE, MAPE and *skill vs naive*, per currency and per horizon.
- 🤖 A scheduled GitHub Action refreshes data and forecasts every week.

> Educational project. Forecasts are statistical estimates, **not financial advice**. Exchange rates are very close to a random walk, which is why every model is compared with the naive baseline.

## Quick start

```bash
git clone https://github.com/JM181998/exchange-rate-visualizer.git
cd exchange-rate-visualizer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

The first launch downloads the daily history (about a minute, once); afterwards only the missing days are fetched.

### Generate / refresh forecasts

The forecast and metrics pages read the files in `outputs/`, produced by the pipeline:

```bash
pip install -r requirements-train.txt          # adds TensorFlow/Keras (Python 3.11 or 3.12)
python -m fxviz.pipeline                       # update data -> backtest -> forecast (naive + LSTM)

python -m fxviz.pipeline --models naive        # baseline only, no TensorFlow needed
python -m fxviz.pipeline --no-update           # use the cached CSV without calling the API
```

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## How it works

| Step | What happens |
|------|--------------|
| Data | `fxviz/data.py` downloads daily rates, shrinking request windows if the API returns downsampled (weekly) data, and caches them in `data/historical_rates.csv`. |
| Models | `fxviz/models.py`: `NaiveForecaster` and `LSTMForecaster` (last 60 daily log-returns → next 30 cumulative log-returns). |
| Evaluation | `fxviz/evaluation.py`: models are trained only on data before the last ~2 years, then forecast 30 business days from a new origin every week with no look-ahead. |
| Intervals | Empirical 10–90 % quantiles of each model's own backtest errors, per horizon. |
| Outputs | `outputs/forecasts.csv`, `metrics.csv`, `horizon_errors.csv`, `run_info.json`. |

## Project structure

```
app.py                  Home: historical rates
pages/                  Forecast · Model metrics · Methodology
fxviz/                  data, models, evaluation, pipeline (importable, tested)
tests/                  pytest suite
data/                   cached rates (CSV)
outputs/                forecasts and metrics written by the pipeline
.github/workflows/      CI (tests) and weekly refresh
```

## Deploy (Streamlit Community Cloud)

1. Push the repo to GitHub.
2. In [share.streamlit.io](https://share.streamlit.io) create an app from the repo with `app.py` as the main file. Only `requirements.txt` is installed, so the app stays light.
3. The weekly **Refresh data and forecasts** workflow commits new data and outputs, and Streamlit redeploys on every push.

## What changed in v2

The first version had issues that this rewrite fixes:

- Long API ranges are returned **weekly**, but the data was treated as daily, so "30 days" was really 30 weeks. Data is now daily and forecasts are in business days with real dates.
- Metrics compared the *future* forecast with the *last 30 known days*, which made R² meaningless. Evaluation is now a proper out-of-sample rolling backtest against a naive baseline.
- The LSTM now models returns instead of MinMax-scaled levels, and is trained with a time-ordered validation split.
- Moved to the current API host, removed in-app training (it does not belong in a deployed app), added tests, CI, a scheduled refresh, prediction intervals and a lighter app install.

## Roadmap

- Migrate the data layer from Frankfurter v1 (deprecated but still available) to v2.
- More currencies and a currency selector for the pipeline.
- Extra baselines (ARIMA / gradient boosting) to compare with the LSTM.

## Author

Juan Manuel Fuentes · [LinkedIn](https://www.linkedin.com/in/juanma-fuentes/) · Released under the [MIT License](LICENSE).
