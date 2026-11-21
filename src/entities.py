"""Map headlines to the US-listed companies they talk about.

A hand-curated alias table is used instead of a generic NER model because
precision matters more than recall here: a headline wrongly attributed to a
stock is pure noise in the return analysis.  Ambiguous names are either
qualified ("Delta Air", "United Airlines") or left out ("Target", "Visa",
"Gap", "Shell").  Matching is case-sensitive with word boundaries, so
"Intel" does not fire on "intelligence" and "Apple" does not fire on "apple".

Companies that were delisted / acquired before today (Twitter, Celgene,
Raytheon pre-merger ...) are excluded because Yahoo no longer serves their
history - see the survivorship-bias note in the README.
"""
from __future__ import annotations

import re

import pandas as pd

# ticker -> list of regex aliases (case-sensitive)
ALIASES: dict[str, list[str]] = {
    # Technology & communication
    "AAPL": [r"Apple(?!\s+(?:Daily|pie))", r"iPhones?"],
    "MSFT": [r"Microsoft"],
    "AMZN": [r"Amazon(?!\s+(?:rainforest|forest|basin|river|fires?))(?:\.com)?", r"Whole Foods"],
    "GOOGL": [r"Google", r"Alphabet", r"YouTube"],
    "META": [r"Facebook", r"Instagram", r"WhatsApp"],
    "NFLX": [r"Netflix"],
    "NVDA": [r"Nvidia"],
    "INTC": [r"Intel"],
    "AMD": [r"AMD", r"Advanced Micro Devices"],
    "QCOM": [r"Qualcomm"],
    "MU": [r"Micron"],
    "AVGO": [r"Broadcom"],
    "IBM": [r"IBM"],
    "ORCL": [r"Oracle(?!\s+of\s+Omaha)"],
    "CSCO": [r"Cisco"],
    "CRM": [r"Salesforce"],
    "ADBE": [r"Adobe"],
    "PYPL": [r"PayPal"],
    "MA": [r"Mastercard"],
    "EBAY": [r"eBay"],
    "HPQ": [r"HP Inc"],
    "DELL": [r"Dell"],
    "ZM": [r"Zoom Video", r"Zoom(?=['’]s| shares| stock)"],
    "UBER": [r"Uber"],
    "LYFT": [r"Lyft"],
    "BABA": [r"Alibaba"],
    "T": [r"AT&T", r"AT&amp;T"],
    "VZ": [r"Verizon"],
    "CMCSA": [r"Comcast", r"NBCUniversal"],
    "DIS": [r"Disney"],
    # Consumer
    "TSLA": [r"Tesla"],
    "GM": [r"General Motors", r"\bGM(?=['’]s|\s+(?:shares|stock|to|will|said|says|recalls?|cuts?|plans?|CEO|workers|plant))"],
    "F": [r"Ford Motor", r"Ford(?=['’]s|\s+(?:shares|stock|to|will|said|says|recalls?|cuts?|plans?))"],
    "WMT": [r"Walmart", r"Wal-Mart"],
    "COST": [r"Costco"],
    "HD": [r"Home Depot"],
    "MCD": [r"McDonald['’]s"],
    "SBUX": [r"Starbucks"],
    "NKE": [r"Nike"],
    "KO": [r"Coca-Cola", r"Coke"],
    "PEP": [r"PepsiCo", r"Pepsi"],
    "PG": [r"Procter & Gamble", r"Procter &amp; Gamble", r"P&G"],
    "KHC": [r"Kraft Heinz"],
    "MO": [r"Altria"],
    "PM": [r"Philip Morris"],
    "MAR": [r"Marriott"],
    "CCL": [r"Carnival Corp", r"Carnival(?=['’]s|\s+(?:Cruise|cruise|shares|stock|said|says|CEO))"],
    "EXPE": [r"Expedia"],
    "CVS": [r"CVS"],
    # Industrials & transport
    "BA": [r"Boeing", r"737 MAX"],
    "GE": [r"General Electric", r"\bGE(?=['’]s|\s)"],
    "CAT": [r"Caterpillar"],
    "MMM": [r"3M"],
    "HON": [r"Honeywell"],
    "LMT": [r"Lockheed"],
    "FDX": [r"FedEx"],
    "UPS": [r"UPS(?!\s+and\s+downs)"],
    "DAL": [r"Delta Air(?:\s?Lines)?"],
    "UAL": [r"United Airlines", r"United Continental"],
    "AAL": [r"American Airlines"],
    # Financials
    "JPM": [r"JPMorgan", r"JP Morgan"],
    "GS": [r"Goldman"],
    "MS": [r"Morgan Stanley"],
    "C": [r"Citigroup", r"Citi(?:bank)?"],
    "BAC": [r"Bank of America", r"BofA"],
    "WFC": [r"Wells Fargo"],
    "BRK-B": [r"Berkshire Hathaway", r"Warren Buffett"],
    "BLK": [r"BlackRock"],
    # Energy
    "XOM": [r"Exxon"],
    "CVX": [r"Chevron"],
    # Healthcare
    "PFE": [r"Pfizer"],
    "JNJ": [r"Johnson & Johnson", r"Johnson &amp; Johnson", r"J&J"],
    "MRK": [r"Merck"],
    "MRNA": [r"Moderna"],
    "GILD": [r"Gilead", r"remdesivir"],
    "ABBV": [r"AbbVie"],
    "BMY": [r"Bristol-Myers", r"Bristol Myers"],
    "LLY": [r"Eli Lilly"],
    "UNH": [r"UnitedHealth"],
}

_PATTERNS = {t: re.compile(r"(?<![\w&])(?:" + "|".join(a) + r")(?![\w])") for t, a in ALIASES.items()}


def tickers() -> list[str]:
    return list(ALIASES)


def find_tickers(text: str) -> list[str]:
    return [t for t, p in _PATTERNS.items() if p.search(text)]


def explode_mentions(headlines: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (headline, ticker) mention."""
    m = headlines.assign(ticker=headlines["text"].map(find_tickers))
    m = m.explode("ticker").dropna(subset=["ticker"])
    return m.reset_index(drop=True)
