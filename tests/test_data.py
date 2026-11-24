import numpy as np
import pandas as pd
import pytest
import requests

from fxviz import config, data


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload


class FakeAPI:
    """Mimics Frankfurter v1: daily rows up to ``daily_limit_days``, weekly above it."""

    def __init__(self, last_date="2025-06-30", daily_limit_days=90, fail=None):
        self.calls, self.last_date = [], pd.Timestamp(last_date)
        self.daily_limit_days, self.fail = daily_limit_days, fail

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if self.fail:
            raise self.fail
        span = url.rsplit("/", 1)[1]
        start, end = (pd.Timestamp(x) for x in span.split(".."))
        end = min(end, self.last_date)
        days = pd.bdate_range(start, end) if start <= end else []
        if (end - start).days > self.daily_limit_days:
            days = [d for d in days if d.weekday() == 0]  # weekly
        rates = {f"{d:%Y-%m-%d}": {c: 1.0 + d.dayofyear / 1000 for c in config.CURRENCIES} for d in days}
        return FakeResponse({"base": "EUR", "rates": rates})


def test_fetch_rates_returns_daily_even_if_api_downsamples_long_ranges():
    api = FakeAPI()
    df = data.fetch_rates("2020-01-01", "2025-06-30", session=api, pause=0, backoff=0)
    assert list(df.columns) == list(config.PAIRS)
    assert df.index.is_monotonic_increasing and not df.index.has_duplicates
    expected = pd.bdate_range("2020-01-01", "2025-06-30")
    assert len(df) == len(expected)  # every business day, not one per week
    assert df.index.to_series().diff().dt.days.median() == 1


def test_fetch_rates_uses_big_windows_when_api_is_daily():
    api = FakeAPI(daily_limit_days=10_000)
    data.fetch_rates("2020-01-01", "2022-12-31", session=api, pause=0)
    assert len(api.calls) == 4  # 1096 days / 365-day windows, instead of ~14 small requests


def test_fetch_rates_raises_data_unavailable_on_network_error():
    api = FakeAPI(fail=requests.ConnectionError("boom"))
    with pytest.raises(data.DataUnavailable):
        data.fetch_rates("2024-01-01", "2024-02-01", session=api, retries=1, backoff=0)


def test_http_client_error_is_not_retried():
    class Bad:
        calls = 0

        def get(self, *a, **k):
            Bad.calls += 1
            return FakeResponse({}, status=400)

    with pytest.raises(data.DataUnavailable):
        data.fetch_rates("2024-01-01", "2024-02-01", session=Bad(), retries=3, backoff=0)
    assert Bad.calls == 1


def test_save_and_load_roundtrip(tmp_path, synthetic_rates):
    path = tmp_path / "rates.csv"
    data.save_rates(synthetic_rates, path)
    loaded = data.load_rates(path)
    pd.testing.assert_frame_equal(loaded, synthetic_rates, check_freq=False, atol=1e-9)


def test_load_rates_missing_file_gives_empty_frame(tmp_path):
    df = data.load_rates(tmp_path / "nope.csv")
    assert df.empty and list(df.columns) == list(config.PAIRS)


def test_is_daily_detects_legacy_weekly_data(synthetic_rates):
    assert data.is_daily(synthetic_rates)
    assert not data.is_daily(synthetic_rates.iloc[::5])


def test_update_downloads_everything_when_cache_is_missing_or_weekly(tmp_path, synthetic_rates):
    path = tmp_path / "rates.csv"
    calls = []

    def fetcher(start, **_):
        calls.append(str(start))
        return synthetic_rates

    data.update_rates(path, fetcher=fetcher)
    assert calls == [config.START_DATE]

    data.save_rates(synthetic_rates.iloc[::5], path)  # legacy weekly cache
    data.update_rates(path, fetcher=fetcher)
    assert calls == [config.START_DATE, config.START_DATE]


def test_update_only_fetches_the_tail_and_merges(tmp_path, synthetic_rates):
    path = tmp_path / "rates.csv"
    old = synthetic_rates.iloc[:-10]
    data.save_rates(old, path)
    seen = {}

    def fetcher(start, **_):
        seen["start"] = pd.Timestamp(start)
        return synthetic_rates.loc[synthetic_rates.index >= start]

    merged = data.update_rates(path, fetcher=fetcher)
    assert seen["start"] == old.index.max() - pd.Timedelta(days=7)
    assert len(merged) == len(synthetic_rates) and not merged.index.has_duplicates
    assert data.load_rates(path).index.max() == synthetic_rates.index.max()


def test_update_keeps_working_when_file_cannot_be_written(tmp_path, synthetic_rates, monkeypatch):
    def boom(*a, **k):
        raise OSError("read-only")

    monkeypatch.setattr(data, "save_rates", boom)
    out = data.update_rates(tmp_path / "x.csv", fetcher=lambda start, **_: synthetic_rates)
    assert len(out) == len(synthetic_rates)


def test_update_raises_when_api_returns_nothing(tmp_path):
    empty = data._payload_to_frame({}, config.CURRENCIES)
    with pytest.raises(data.DataUnavailable):
        data.update_rates(tmp_path / "x.csv", fetcher=lambda start, **_: empty)
