"""Project-wide constants. Paths are anchored to the repo root, so the app and the
pipeline work no matter which directory they are launched from."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "historical_rates.csv"
OUTPUT_DIR = ROOT / "outputs"

# Frankfurter serves the ECB euro foreign exchange reference rates.
# v1 is deprecated in favour of v2 but the maintainers state it stays available.
API_URL = "https://api.frankfurter.dev/v1"
START_DATE = "1999-01-04"  # first ECB reference rate

CURRENCIES = ("USD", "AUD", "GBP", "CHF")


def pair(code: str) -> str:
    """Column name for a currency. 'USD/EUR' means: how many USD per 1 EUR."""
    return f"{code}/EUR"


PAIRS = tuple(pair(c) for c in CURRENCIES)

# Forecasting / evaluation setup (all in business days, the ECB does not publish on weekends)
HORIZON = 30
LOOKBACK = 60
TEST_SIZE = 500  # ~2 years held out for the backtest
STEP = 5  # a new forecast origin every week
INTERVAL = (0.10, 0.90)  # 80 % empirical prediction interval
