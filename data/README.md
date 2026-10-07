# QQQ / TQQQ / SQQQ, 21–25 June 2024 (data snapshot)

File: `qqq-tqqq-sqqq-2024-06-21-25.csv`: three trading dates, one column per fund.

Used by `notebooks/leveraged-etf-volatility-drag.ipynb` (Section 7) to produce `figures/05-qqq-tqqq-sqqq-roundtrip.png`. The notebook reads this file and needs no internet access.

## What the numbers are

| Item | Value |
|---|---|
| Data source | Yahoo Finance, through the `yfinance` Python package |
| Price field | `Close` as returned by `yfinance.download(...)` with its then-default `auto_adjust=True`, i.e. an **adjusted** close (adjusted for splits and distributions) |
| Provenance | Recovered from the saved output of the author's original Colab notebook (`leverage testing.ipynb`). The three dates were *not* freshly downloaded when this repository was created |
| Retrieval date | **Not recorded.** The date of the original Colab run was not preserved, and no fresh retrieval was possible when the snapshot was committed |
| Precision | Six decimals, as printed by the original output |
| Fund roles | TQQQ targets +3× and SQQQ targets −3× the **daily** return of the Nasdaq-100 Index. QQQ is a tradable ETF on the same index and is used here only as a convenient proxy, not as the contractual benchmark |

## Why QQQ's near-flat result is dividend-adjusted

QQQ went ex-dividend on 24 June 2024 (a distribution of 0.7615 US dollars per share), inside the interval. An independently retrieved series of *unadjusted* closes (provider not recorded here) was 480.18, 473.96 and 479.38 for 21, 24 and 25 June, which gives a raw price return of about −0.1666% over the interval. Accounting for the distribution gives about −0.008%, matching the adjusted series in the CSV.

So "QQQ is essentially flat" is a **dividend-adjusted return statement, not an unadjusted price round trip.** The independent series also reproduces the TQQQ and SQQQ endpoint returns to rounding (about −0.161% and −0.119%). Absolute TQQQ and SQQQ levels are affected by later split adjustments and differ between providers and conventions, which is why the analysis uses only ratios within the window.

## Selection caveat

The sequence was originally found by an exploratory search of QQQ history for a short fall-and-rebound. It is therefore a **selected illustration of a mechanism**, not evidence about how often such sequences occur or about the expected performance of leveraged ETFs.

## Re-validating

The notebook has an optional validation cell that, when Yahoo Finance is reachable, fetches the same dates and compares **returns** (not price levels) with this snapshot. It prints the differences and never fails the run.

## Terms of use

The values originate from Yahoo Finance; they are redistributed here only as a three-row snapshot for verification of the analysis. The repository's MIT license covers the code and notebook, not third-party market data.


## Five-year IBKR daily history

For the real-data revision of the article, the repository also contains five years of daily regular-session OHLCV bars retrieved through Interactive Brokers on 2026-10-07:

- `ibkr-qqq-daily-5y.csv`
- `ibkr-tqqq-daily-5y.csv`
- `ibkr-sqqq-daily-5y.csv`

Source: Interactive Brokers historical price history, daily bars, regular trading hours only (`outside_rth=false`), with corporate-action metadata requested during retrieval. The returned chart source was `Last`.

These files are intended for empirical path/window searches and rolling historical analysis. They should be treated as **observed price-bar data**, not as a total-return series unless adjustment conventions are independently established. For this revision it is acceptable to search for new examples in raw price-return space rather than force the earlier dividend-adjusted June 2024 example into every analysis.

The current QQQ file spans 2021-10-11 through 2026-10-07. TQQQ and SQQQ were retrieved using the same five-year request. Exact row counts and overlap should be checked by the analysis notebook before use.
