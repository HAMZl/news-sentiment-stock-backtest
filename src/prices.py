"""Daily split/dividend-adjusted open & close prices from Yahoo Finance (cached)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "cache"
BENCHMARK = "SPY"


def load_prices(tickers: list[str], start="2017-11-01", end="2020-10-31") -> dict[str, pd.DataFrame]:
    """Return {'open': DataFrame, 'close': DataFrame} indexed by trading day, one column per ticker."""
    CACHE.mkdir(exist_ok=True)
    syms = sorted(set(tickers) | {BENCHMARK})
    path = CACHE / f"prices_{start}_{end}_{len(syms)}.pkl"
    if path.exists():
        return pd.read_pickle(path)

    import yfinance as yf

    raw = yf.download(syms, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    px = {"open": raw["Open"][syms], "close": raw["Close"][syms]}
    missing = [s for s in syms if px["close"][s].isna().all()]
    if missing:
        print(f"  no price history for: {missing}")
    pd.to_pickle(px, path)
    return px
