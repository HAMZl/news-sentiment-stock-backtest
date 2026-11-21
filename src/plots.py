"""Static figures for the README (matplotlib)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
POSITIVE = "#2a78d6"   # blue
NEGATIVE = "#eb6834"   # orange
NEUTRAL = "#9a9890"    # gray
BENCH = "#52514e"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False, "axes.axisbelow": True,
})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def market_sentiment(daily_sent: pd.Series, spy: pd.Series, path: Path, label: str):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    s = daily_sent.rolling(20, min_periods=5).mean()
    a1.plot(s.index, s.values, color=POSITIVE)
    a1.axhline(0, color=INK_2, lw=0.8)
    a1.set_title(f"Average {label} sentiment of all headlines (20-day rolling mean)")
    a2.plot(spy.index, spy.values, color=BENCH)
    a2.set_title("S&P 500 (SPY, dividend-adjusted)")
    for a in (a1, a2):
        a.axvspan(pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"), color=NEGATIVE, alpha=0.10, lw=0)
    a2.text(pd.Timestamp("2020-02-14"), spy.max(), "COVID crash", color=INK_2, fontsize=8, va="top", ha="right")
    a2.set_xlim(s.dropna().index.min(), s.index.max())
    _save(fig, path)


def lead_lag(tables: dict[str, pd.DataFrame], path: Path):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    names = list(tables)
    width = 0.38
    colors = [INK, NEUTRAL]
    for i, n in enumerate(names):
        t = tables[n]
        x = t["lag"].to_numpy() + (i - (len(names) - 1) / 2) * width
        ax.bar(x, t["corr"], width=width - 0.04, color=colors[i], label=n)
        for xi, c, tv in zip(x, t["corr"], t["t"]):
            if abs(tv) > 2:
                ax.text(xi, c + (0.012 if c >= 0 else -0.012), "*", ha="center",
                        va="bottom" if c >= 0 else "top", color=INK_2)
    ax.axhline(0, color=INK_2, lw=0.8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo * 1.15, hi * 1.2)
    ax.set_xticks(range(-3, 4))
    ax.set_xticklabels(["t-3", "t-2", "t-1", "same day", "t+1", "t+2", "t+3"])
    ax.set_ylabel("correlation with S&P 500 daily return")
    ax.set_title("News sentiment follows the market; it does not lead it   (* = |t| > 2, HAC)")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def car_plot(paths: dict[str, pd.DataFrame], path: Path):
    fig, axes = plt.subplots(1, len(paths), figsize=(10, 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, cp) in zip(axes, paths.items()):
        for b, c in [("positive", POSITIVE), ("neutral", NEUTRAL), ("negative", NEGATIVE)]:
            ax.plot(cp.index, cp[b] * 100, color=c, label=b)
            ax.text(cp.index[-1] + 0.3, cp[b].iloc[-1] * 100, b, color=INK_2, fontsize=8, va="center")
        ax.axvspan(-0.5, 1.5, color=INK_2, alpha=0.07, lw=0)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("trading days relative to the news date")
        ax.set_xlim(cp.index[0], cp.index[-1] + 3)
    axes[0].set_ylabel("cumulative abnormal return vs SPY (%)")
    axes[0].text(0.5, axes[0].get_ylim()[1] * 0.92, "news\nwindow", ha="center", fontsize=8, color=INK_2, va="top")
    fig.suptitle("Most of the price move happens before and on the news day",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, path)


def equity_curves(rets: pd.DataFrame, path: Path, title: str):
    styles = {
        "Positive sentiment (long)": dict(color=POSITIVE),
        "Negative sentiment (long)": dict(color=NEGATIVE),
        "S&P 500 (SPY)": dict(color=BENCH),
        "Universe equal-weight": dict(color=NEUTRAL, ls="--"),
    }
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ends = []
    for k, st in styles.items():
        eq = (1 + rets[k]).cumprod() * 100
        ax.plot(eq.index, eq.values, label=k, **st)
        ends.append((eq.iloc[-1], k))
    # direct labels, nudged apart so they don't collide
    ends.sort()
    last_y = -np.inf
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for y, k in ends:
        y_lab = max(y, last_y + span * 0.055)
        ax.text(rets.index[-1] + pd.Timedelta(days=6), y_lab, f"{k}  {y - 100:+.0f}%", fontsize=8,
                color=INK_2, va="center")
        last_y = y_lab
    ax.axhline(100, color=INK_2, lw=0.8)
    ax.set_xlim(rets.index[0], rets.index[-1] + pd.Timedelta(days=150))
    ax.set_ylabel("growth of $100 (after 10 bps costs)")
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, path)
