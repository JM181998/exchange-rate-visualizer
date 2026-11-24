"""Download, cache and incrementally update the ECB reference rates (via Frankfurter)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

# Frankfurter downsamples long ranges to weekly values. We start with large windows and
# shrink them if the API answers with coarse data, so we always end up with daily rates.
CHUNK_STEPS_DAYS = (365, 80, 28)


class DataUnavailable(RuntimeError):
    """The rates API could not be reached or returned something unusable."""


def _get_json(session, url: str, params: dict, timeout: float, retries: int, backoff: float) -> dict:
    last_error = "unknown error"
    for attempt in range(retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
            else:
                raise DataUnavailable(f"Frankfurter API returned HTTP {response.status_code} for {url}")
        except requests.RequestException as exc:
            last_error = str(exc)
        except ValueError:
            last_error = "invalid JSON in response"
        if attempt < retries:
            time.sleep(backoff * 2**attempt)
    raise DataUnavailable(f"Frankfurter API unreachable ({last_error})")


def _payload_to_frame(payload: dict, currencies) -> pd.DataFrame:
    rates = payload.get("rates") or {}
    dates = sorted(rates)
    rows = [[rates[d].get(c, np.nan) for c in currencies] for d in dates]
    frame = pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(pd.to_datetime(dates), name="Date"),
        columns=[config.pair(c) for c in currencies],
        dtype=float,
    )
    return frame


def fetch_rates(
    start,
    end=None,
    currencies=config.CURRENCIES,
    base_url: str = config.API_URL,
    session=None,
    timeout: float = 30,
    retries: int = 3,
    backoff: float = 1.0,
    pause: float = 0.05,
) -> pd.DataFrame:
    """Daily EUR reference rates for ``start..end`` (inclusive). Index: business days."""
    session = session or requests.Session()
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end if end is not None else pd.Timestamp.today()).normalize()
    symbols = ",".join(currencies)

    frames: list[pd.DataFrame] = []
    cursor, step_idx = start, 0
    while cursor <= end:
        window_end = min(cursor + pd.Timedelta(days=CHUNK_STEPS_DAYS[step_idx] - 1), end)
        url = f"{base_url}/{cursor:%Y-%m-%d}..{window_end:%Y-%m-%d}"
        payload = _get_json(session, url, {"base": "EUR", "symbols": symbols}, timeout, retries, backoff)
        frame = _payload_to_frame(payload, currencies)

        span_days = (window_end - cursor).days
        expected = len(pd.bdate_range(cursor, window_end))
        too_coarse = span_days >= 20 and len(frame) < 0.5 * expected
        if too_coarse and step_idx < len(CHUNK_STEPS_DAYS) - 1:
            step_idx += 1  # same window again, but smaller
            continue

        frames.append(frame.loc[frame.index >= cursor])
        cursor = window_end + pd.Timedelta(days=1)
        if pause:
            time.sleep(pause)

    if not frames:
        return _payload_to_frame({}, currencies)
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="last")].sort_index()


def load_rates(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Read the cached CSV (empty frame if it does not exist yet)."""
    path = Path(path)
    if not path.exists():
        return _payload_to_frame({}, config.CURRENCIES)
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df = df.apply(pd.to_numeric, errors="coerce").sort_index()
    return df[[c for c in config.PAIRS if c in df.columns]]


def save_rates(df: pd.DataFrame, path: Path = config.DATA_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="Date")


def is_daily(df: pd.DataFrame) -> bool:
    """False for the legacy weekly dataset (median gap between observations > 3 days)."""
    if len(df) < 3:
        return True
    return df.index.to_series().diff().dt.days.median() <= 3


def update_rates(path: Path = config.DATA_PATH, save: bool = True, fetcher=fetch_rates) -> pd.DataFrame:
    """Bring the cached dataset up to date and return it.

    * no cache, or a legacy weekly cache -> full daily download since 1999
    * otherwise -> only the missing tail (re-fetching the last 7 days, since the most
      recent values can be revised)
    Raises ``DataUnavailable`` if the API cannot be used; callers can fall back to ``load_rates``.
    """
    existing = load_rates(path)
    if existing.empty or not is_daily(existing):
        merged = fetcher(config.START_DATE)
    else:
        tail = fetcher(existing.index.max() - pd.Timedelta(days=7))
        merged = pd.concat([existing, tail])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    if merged.empty:
        raise DataUnavailable("The API returned no data.")
    if save:
        try:
            save_rates(merged, path)
        except OSError as exc:  # read-only deployments: keep working in memory
            log.warning("Could not write %s: %s", path, exc)
    return merged
