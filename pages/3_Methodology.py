import streamlit as st

st.set_page_config(page_title="Methodology · Exchange Rate Visualizer", page_icon="🧪", layout="wide")

st.title("🧪 Methodology")
st.markdown(
    """
### Data
Daily euro foreign-exchange reference rates from the **European Central Bank**, downloaded from the
[Frankfurter API](https://frankfurter.dev) (1999 → today, business days only). `USD/EUR` means
*how many USD for 1 EUR*. The dataset is cached in `data/historical_rates.csv` and updated
incrementally: only the missing days are downloaded.

### Models
- **Naive (random walk)**: every future day equals the last known rate. For exchange rates this
  is a famously strong baseline, so it is the reference for everything else.
- **LSTM**: a small LSTM (Keras 3) fed with the last 60 daily log-returns that outputs the next
  30 cumulative log-returns at once. Working on returns keeps the series stationary and avoids
  the flat, mean-reverting forecasts that a model on raw price levels tends to produce.

### Evaluation
Rolling-origin backtest on the last ~2 years: the model is trained only on data *before* that
window, then forecasts 30 business days ahead from a new origin every week, using only the data
available at that origin. Metrics (MAE, RMSE, MAPE) are computed on those out-of-sample forecasts
and compared with the naive baseline (*skill score*). Forecast origins overlap, so the
numbers are indicative rather than statistically independent.

### Prediction intervals
The shaded band is the empirical 10–90 % range of each model's *own* backtest errors at each horizon,
so it widens with the horizon and is not a Gaussian assumption.

### Limitations
Exchange rates react to news, central-bank decisions and shocks that no price-history model can see.
Expect forecasts to be close to the last rate, with most of the value in the uncertainty band.
**This project is educational and not financial advice.**
"""
)
