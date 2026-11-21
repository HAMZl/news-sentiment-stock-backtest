"""Core analyses: event study, market lead/lag, trading backtest, text classifier.

Timing convention (important - this is where most sentiment "alphas" leak):
Reuters and Guardian only carry a calendar date, not a timestamp, so a
headline dated D may have appeared before, during or after the US session.

* ``t0`` = first trading day >= D  (the day the news is *in* prices)
* ``t1`` = first trading day  > D  (the first day we could trade on it
  without look-ahead: enter at the OPEN of t1)

Anything measured from close(t0-1) to close(t0) is a *reaction* and is
reported only to show how much the market already knew.  Every "can you
make money" number uses entries at open(t1) or later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

POS, NEG = 0.3, -0.3          # sentiment bucket thresholds (fixed, not tuned)
HORIZONS = (1, 5, 10, 20)     # holding periods in trading days
COST_BPS = 10                 # one-way transaction cost, basis points


# ---------------------------------------------------------------- helpers
def attach_trading_days(df: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    d = df["date"].values.astype("datetime64[ns]")
    out = df.copy()
    out["t0"] = np.searchsorted(days.values, d, side="left")
    out["t1"] = np.searchsorted(days.values, d, side="right")
    return out


def bucket(s: pd.Series) -> pd.Series:
    return pd.Series(np.select([s > POS, s < NEG], ["positive", "negative"], "neutral"), index=s.index)


def clustered_mean(y: pd.Series, groups: pd.Series) -> dict:
    """Mean of y with standard error clustered by `groups` (e.g. event date).

    Events on the same day (e.g. the March-2020 crash) are not independent,
    so naive t-stats would be badly overstated.
    """
    y = pd.Series(y).reset_index(drop=True)
    g = pd.Series(groups).reset_index(drop=True)
    ok = y.notna()
    y, g = y[ok], g[ok]
    if len(y) < 10:
        return {"mean": np.nan, "se": np.nan, "t": np.nan, "n": int(len(y))}
    res = sm.OLS(y.to_numpy(), np.ones(len(y))).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g)[0]})
    return {"mean": float(res.params[0]), "se": float(res.bse[0]), "t": float(res.tvalues[0]), "n": int(len(y))}


# ---------------------------------------------------------------- events
def build_events(mentions: pd.DataFrame, px: dict, score_col: str) -> pd.DataFrame:
    """One row per (ticker, news date) with mean sentiment and forward returns."""
    close, open_ = px["close"], px["open"]
    days = close.index
    ev = (mentions.groupby(["ticker", "date"])[score_col]
          .agg(sent="mean", n="size").reset_index())
    ev = ev[ev["ticker"].isin(close.columns)]
    ev = attach_trading_days(ev, days)
    ev["bucket"] = bucket(ev["sent"])

    T = len(days)
    C, O = close.to_numpy(), open_.to_numpy()
    col = {c: i for i, c in enumerate(close.columns)}
    spy = col["SPY"]
    j = ev["ticker"].map(col).to_numpy()
    t0, t1 = ev["t0"].to_numpy(), ev["t1"].to_numpy()

    def ret(a_idx, a_px, b_idx, b_px, jj):
        ok = (a_idx >= 0) & (b_idx < T) & (a_idx < T)
        a_i, b_i = np.clip(a_idx, 0, T - 1), np.clip(b_idx, 0, T - 1)
        r = b_px[b_i, jj] / a_px[a_i, jj] - 1
        return np.where(ok, r, np.nan)

    # Reaction: close(t0-1) -> close(t0)
    react = ret(t0 - 1, C, t0, C, j) - ret(t0 - 1, C, t0, C, np.full_like(j, spy))
    ev["ar_reaction"] = react
    # Tradeable: open(t1) -> close(t1+h-1), market-adjusted
    for h in HORIZONS:
        r_s = ret(t1, O, t1 + h - 1, C, j)
        r_m = ret(t1, O, t1 + h - 1, C, np.full_like(j, spy))
        ev[f"ar_fwd_{h}d"] = r_s - r_m
    return ev.dropna(subset=["ar_reaction"])


def event_study_table(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = ["ar_reaction"] + [f"ar_fwd_{h}d" for h in HORIZONS]
    for b in ["positive", "neutral", "negative"]:
        sub = ev[ev["bucket"] == b]
        for c in cols:
            r = clustered_mean(sub[c], sub["date"])
            rows.append({"bucket": b, "window": c, **r})
    # Positive minus negative spread via regression on a dummy, clustered by date
    pn = ev[ev["bucket"] != "neutral"]
    for c in cols:
        d = pn.dropna(subset=[c])
        X = sm.add_constant((d["bucket"] == "positive").astype(float).to_numpy())
        res = sm.OLS(d[c].to_numpy(), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["date"])[0]})
        rows.append({"bucket": "pos - neg", "window": c, "mean": res.params[1], "se": res.bse[1],
                     "t": res.tvalues[1], "n": len(d)})
    t = pd.DataFrame(rows)
    t["mean_bps"] = t["mean"] * 1e4
    return t


def car_paths(ev: pd.DataFrame, px: dict, pre=5, post=10) -> pd.DataFrame:
    """Cumulative abnormal (close-to-close, SPY-adjusted) return around t0, by bucket."""
    close = px["close"]
    r = close.pct_change()
    ar = r.sub(r["SPY"], axis=0).to_numpy()
    col = {c: i for i, c in enumerate(close.columns)}
    T = len(close)
    ks = np.arange(-pre, post + 1)
    j = ev["ticker"].map(col).to_numpy()
    idx = ev["t0"].to_numpy()[:, None] + ks[None, :]
    ok = (idx >= 1) & (idx < T)
    vals = np.where(ok, ar[np.clip(idx, 0, T - 1), j[:, None]], np.nan)
    out = {}
    for b in ["positive", "neutral", "negative"]:
        m = (ev["bucket"] == b).to_numpy()
        out[b] = np.nancumsum(np.nanmean(vals[m], axis=0))
    return pd.DataFrame(out, index=ks)


# ---------------------------------------------------------------- market level
def market_lead_lag(headlines: pd.DataFrame, px: dict, score_col: str, lags=range(-3, 4)) -> pd.DataFrame:
    """Correlation of daily mean headline sentiment (all news) with SPY returns at various lags.

    lag k < 0: past returns (does sentiment *follow* the market?)
    lag k = 0: same-day return
    lag k > 0: future returns (does sentiment *lead* the market?)
    """
    close = px["close"]["SPY"]
    days = close.index
    h = attach_trading_days(headlines[["date", score_col]], days)
    h = h[h["t0"] < len(days)]
    daily = h.groupby("t0")[score_col].mean()
    r = close.pct_change().to_numpy()
    rows = []
    for k in lags:
        idx = daily.index.to_numpy() + k
        ok = (idx >= 1) & (idx < len(days))
        x, y = daily.to_numpy()[ok], r[idx[ok]]
        x = (x - x.mean()) / x.std()
        res = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        rows.append({"lag": k, "corr": float(np.corrcoef(x, y)[0, 1]), "beta_bps_per_sd": res.params[1] * 1e4,
                     "t": float(res.tvalues[1]), "n": int(ok.sum())})
    return pd.DataFrame(rows)


def daily_market_sentiment(headlines: pd.DataFrame, score_col: str) -> pd.Series:
    return headlines.groupby("date")[score_col].mean()


# ---------------------------------------------------------------- backtest
def backtest(ev: pd.DataFrame, px: dict, hold: int = 5, cost_bps: float = COST_BPS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily-rebalanced, equal-weight portfolios built from news-day signals.

    A stock enters at the OPEN of t1 (first session after the news date) and is
    held for `hold` sessions.  If more news arrives, the window is refreshed
    and the signal is the average sentiment of all headlines in the window.
    """
    close, open_ = px["close"], px["open"]
    days = close.index
    names = [c for c in close.columns if c != "SPY"]
    T = len(days)
    e = ev[(ev["t1"] < T) & ev["ticker"].isin(names)]
    s_sum = (e.pivot_table(index="t1", columns="ticker", values="sent", aggfunc="sum")
             .reindex(index=range(T), columns=names).fillna(0.0))
    s_cnt = (e.pivot_table(index="t1", columns="ticker", values="sent", aggfunc="count")
             .reindex(index=range(T), columns=names).fillna(0.0))
    win_sum = s_sum.rolling(hold, min_periods=1).sum().to_numpy()
    win_cnt = s_cnt.rolling(hold, min_periods=1).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(win_cnt > 0, win_sum / win_cnt, np.nan)

    C = close[names].to_numpy()
    O = open_[names].to_numpy()
    cc = np.vstack([np.full((1, len(names)), np.nan), C[1:] / C[:-1] - 1])
    oc = C / O - 1
    tradable = ~np.isnan(C) & ~np.isnan(O)

    def run(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = pos & tradable
        prev = np.vstack([np.zeros((1, pos.shape[1]), bool), pos[:-1]])
        r = np.where(prev, cc, oc)
        r = np.where(pos, r, 0.0)
        n = pos.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            gross = np.where(n > 0, np.nansum(r, 1) / np.maximum(n, 1), 0.0)
            w = np.where(n[:, None] > 0, pos / np.maximum(n, 1)[:, None], 0.0)
        w_prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
        turnover = np.abs(w - w_prev).sum(1)
        return gross - turnover * cost_bps / 1e4, n

    sig = {
        "Positive sentiment (long)": S > POS,
        "Negative sentiment (long)": S < NEG,
        "Any news (long)": win_cnt > 0,
    }
    rets, counts = {}, {}
    for k, m in sig.items():          # NaN comparisons are False, so masks are clean bools
        rets[k], counts[k] = run(m)
    rets["Long positive / short negative"] = rets["Positive sentiment (long)"] - rets["Negative sentiment (long)"]
    with np.errstate(invalid="ignore"):
        rets["Universe equal-weight"] = np.nanmean(np.where(tradable, cc, np.nan), 1)
    rets["S&P 500 (SPY)"] = close["SPY"].pct_change().to_numpy()

    out = pd.DataFrame(rets, index=days).fillna(0.0)
    # Restrict to the period where news is flowing
    first = days[int(e["t1"].min())]
    last = days[min(int(e["t1"].max()) + hold, T - 1)]
    out = out.loc[first:last]
    holdings = pd.DataFrame(counts, index=days).loc[first:last]
    return out, holdings


def perf_table(rets: pd.DataFrame, holdings: pd.DataFrame | None = None) -> pd.DataFrame:
    spy = rets["S&P 500 (SPY)"]
    rows = []
    for k in rets:
        r = rets[k]
        eq = (1 + r).cumprod()
        yrs = len(r) / 252
        res = sm.OLS(r.to_numpy(), sm.add_constant(spy.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        row = {
            "strategy": k,
            "total_return": eq.iloc[-1] - 1,
            "cagr": eq.iloc[-1] ** (1 / yrs) - 1,
            "ann_vol": r.std() * np.sqrt(252),
            "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
            "max_drawdown": (eq / eq.cummax() - 1).min(),
            "beta": res.params[1],
            "alpha_ann": res.params[0] * 252,
            "alpha_t": res.tvalues[0],
        }
        if holdings is not None and k in holdings:
            row["avg_holdings"] = holdings[k][holdings[k] > 0].mean()
            row["pct_days_invested"] = (holdings[k] > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def yearly_sharpe(rets: pd.DataFrame) -> pd.DataFrame:
    g = rets.groupby(rets.index.year)
    out = g.apply(lambda d: d.mean() / d.std() * np.sqrt(252))
    out = out[g.size() >= 60]            # skip stub years (e.g. the last week of 2017)
    out.index = [f"{y}" + (" (to Jul)" if y == 2020 else "") for y in out.index]
    return out.T


# ---------------------------------------------------------------- text classifier
def text_classifier(mentions: pd.DataFrame, ev: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Can the headline *text* predict the stock's move?  (Fixed version of the original NB script.)

    Label = sign of the market-adjusted return.  Split is chronological
    (train on the first 70% of dates, test on the last 30%) - a random split
    leaks future information through overlapping stories.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.naive_bayes import MultinomialNB

    m = mentions.merge(ev[["ticker", "date", "ar_reaction", "ar_fwd_1d", "ar_fwd_5d"]], on=["ticker", "date"])
    cutoff = m["date"].quantile(0.7)
    rows = []
    for target, label in [("ar_reaction", "Same-day reaction (not tradeable)"),
                          ("ar_fwd_1d", "Next session, open->close"),
                          ("ar_fwd_5d", "Next 5 sessions")]:
        d = m.dropna(subset=[target])
        d = d[d[target] != 0]
        y = (d[target] > 0).astype(int)
        tr, te = d["date"] < cutoff, d["date"] >= cutoff
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
        Xtr, Xte = vec.fit_transform(d.loc[tr, "text"]), vec.transform(d.loc[te, "text"])
        majority = int(y[tr].mean() >= 0.5)
        models = {
            "Majority class": None,
            "Naive Bayes (TF-IDF)": MultinomialNB(),
            "Logistic regression (TF-IDF)": LogisticRegression(C=1.0, max_iter=2000),
            "Sentiment score only": "score",
        }
        for name, mdl in models.items():
            if mdl is None:
                pred, prob = np.full(te.sum(), majority), np.full(te.sum(), 0.5)
            elif mdl == "score":
                s = d.loc[te, score_col].to_numpy()
                pred, prob = (s > 0).astype(int), s
            else:
                mdl.fit(Xtr, y[tr])
                prob = mdl.predict_proba(Xte)[:, 1]
                pred = (prob > 0.5).astype(int)
            rows.append({"target": label, "model": name,
                         "accuracy": accuracy_score(y[te], pred),
                         "auc": roc_auc_score(y[te], prob) if mdl is not None else 0.5,
                         "n_train": int(tr.sum()), "n_test": int(te.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the "buy list"
def sentiment_screen(ev: pd.DataFrame, px: dict, asof: str, lookback_days=30, min_n=3, top=10, hold=20) -> pd.DataFrame:
    """Rank stocks by average headline sentiment over a trailing window, then
    measure what actually happened over the next `hold` sessions."""
    asof = pd.Timestamp(asof)
    w = ev[(ev["date"] > asof - pd.Timedelta(days=lookback_days)) & (ev["date"] <= asof)]
    g = w.groupby("ticker").apply(lambda d: pd.Series({"sentiment": np.average(d["sent"], weights=d["n"]),
                                                       "headlines": d["n"].sum()}))
    g = g[g["headlines"] >= min_n].sort_values("sentiment", ascending=False)
    close, open_ = px["close"], px["open"]
    t1 = close.index.searchsorted(asof, side="right")
    t2 = min(t1 + hold - 1, len(close) - 1)
    fwd = close.iloc[t2] / open_.iloc[t1] - 1
    g["fwd_return"] = fwd.reindex(g.index)
    g["fwd_excess_vs_spy"] = g["fwd_return"] - fwd["SPY"]
    g["rank"] = np.arange(1, len(g) + 1)
    g.attrs.update(entry=str(close.index[t1].date()), exit=str(close.index[t2].date()))
    return pd.concat([g.head(top).assign(side="top"), g.tail(top).assign(side="bottom")])
