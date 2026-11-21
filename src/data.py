"""Load and clean the three financial-news headline datasets.

Each source ships in a slightly different shape:

* CNBC      - Headlines, Time (" 7:51  PM ET Fri, 17 July 2020"), Description
* Reuters   - Headlines, Time ("Jul 18 2020"), Description
* Guardian  - Time ("18-Jul-20"), Headlines (no description)

They are normalised into one frame with columns
``source, date, headline, description, text``.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_WS = re.compile(r"\s+")


def _clean(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace(_WS, " ", regex=True).str.strip()


def _load_cnbc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["Headlines"])
    # " 7:51  PM ET Fri, 17 July 2020" -> "17 July 2020"
    day = df["Time"].astype(str).str.extract(r"(\d{1,2}\s+\w+\s+\d{4})")[0]
    return pd.DataFrame({
        "source": "cnbc",
        "date": pd.to_datetime(day.str.replace(_WS, " ", regex=True), format="mixed", errors="coerce"),
        "headline": _clean(df["Headlines"]),
        "description": _clean(df["Description"]),
    })


def _load_reuters(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["Headlines"])
    return pd.DataFrame({
        "source": "reuters",
        "date": pd.to_datetime(df["Time"], format="%b %d %Y", errors="coerce"),
        "headline": _clean(df["Headlines"]),
        "description": _clean(df["Description"]),
    })


def _load_guardian(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["Headlines"])
    return pd.DataFrame({
        "source": "guardian",
        "date": pd.to_datetime(df["Time"], format="%d-%b-%y", errors="coerce"),
        "headline": _clean(df["Headlines"]),
        "description": "",
    })


def load_headlines(root: Path = ROOT) -> pd.DataFrame:
    frames = [
        _load_cnbc(root / "cnbc_headlines.csv"),
        _load_reuters(root / "reuters_headlines.csv"),
        _load_guardian(root / "guardian_headlines.csv"),
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["date"])
    df = df[df["headline"].str.len() > 0]
    df = df.drop_duplicates(subset=["source", "date", "headline"])
    # Text used for sentiment and company matching: headline + first-sentence summary.
    df["text"] = (df["headline"] + ". " + df["description"]).str.strip(" .")
    return df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    h = load_headlines()
    print(h.groupby("source")["date"].agg(["count", "min", "max"]))
