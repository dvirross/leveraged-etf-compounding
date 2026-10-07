# Why a daily 3× leveraged product is not 3× the long-term return

Reproducible analysis behind the article **[Why 3× Daily Leverage Is Not 3× Long-Term Return](https://dvirross.com/quant-research/notes/why-3x-daily-leverage-is-not-3x-long-term-return)** on dvirross.com. The article explains the ideas for a general reader; this repository holds the data, code, results and figures so that every number can be checked.

## The question

A daily 3× leveraged ETF targets three times the index's return **each day**. Compounded over many days, the result is not three times the index's cumulative return. How large is the gap, what drives it, and how does it look in real market history?

## What is here

| Part | What it does |
|---|---|
| [`notebooks/leveraged-etf-empirical-analysis.ipynb`](notebooks/leveraged-etf-empirical-analysis.ipynb) | **Source of the article's market-facing figures (2 to 5).** Real QQQ, TQQQ and SQQQ daily history: a real fall-and-recovery, three matched 20-day windows, 63-day rolling windows, positive-day analysis. |
| [`notebooks/leveraged-etf-model-simulations.ipynb`](notebooks/leveraged-etf-model-simulations.ipynb) | Mathematics and **synthetic** simulations only (closed-form two-day example used as article Figure 1, drift-versus-variance experiments, Monte Carlo). Contains no market data and is not used for any market-facing claim. |
| [`analysis/`](analysis) | `empirical.py` (loading with validation, rolling windows, matched-window search, round-trip search), `figures.py` (Plotly figure builders and exporter), `run.py` (regenerates every table and figure). |
| [`results/`](results) | Full-precision CSV outputs and the selected windows. |
| [`figures/`](figures) | Plotly JSON specs (interactive, used by the website) and static PNG fallbacks, plus `figures.json` (alt text, captions, assumptions). `figures/model/` holds the static model figures. |
| [`data/`](data) | Five-year IBKR daily bars for QQQ, TQQQ and SQQQ, and an older three-date snapshot kept for provenance. |
| [`tests/`](tests) | Unit tests of the analysis code on hand-made fixtures (not market data). |

## Data

`data/ibkr-qqq-daily-5y.csv`, `ibkr-tqqq-daily-5y.csv`, `ibkr-sqqq-daily-5y.csv`: 1,253 daily bars each, 2021-10-11 to 2026-10-07, from Interactive Brokers historical price history (daily bars, regular trading hours only, chart source `Last`, retrieved 2026-10-07). They are **observed price bars, not total-return adjusted**. QQQ is a tradable proxy for the Nasdaq-100 that TQQQ and SQQQ target. See [`data/README.md`](data/README.md).

## Method in brief

* **Idealized 3× product:** `value_t = value_{t-1} × (1 + 3 r_t)` applied to QQQ's own daily returns; no fees, financing, tracking error or distributions.
* **Figure 2, round trip:** every two-session window in which QQQ returns to within ±0.10% of its start after a first-day fall; the largest first-day fall is shown (17, 21 and 22 April 2025).
* **Figure 3, matched windows:** all rolling 20-return windows with QQQ return of at least +5%; sets of three non-overlapping windows with the same number of positive days and QQQ returns within a 0.5 percentage-point band; the set with the largest spread of idealized 3× return is shown (2022-06-17 to 2022-07-19, 2023-05-02 to 2023-05-31, 2025-04-03 to 2025-05-02). The rule is fixed in code before the results are inspected, but it deliberately looks for a visible spread, so the windows illustrate a mechanism and are not a random sample. Alternatives and tolerance sensitivity are in `results/`.
* **Figures 4 and 5, rolling windows:** all 1,190 overlapping 63-return windows. Overlapping windows are strongly dependent (about 19 non-overlapping quarters), so the scatter describes this one five-year episode, not 1,190 independent experiments. `results/horizon_sensitivity.csv` shows the 20, 63, 126 and 252-day alternatives.

## Reproduce it

```bash
git clone https://github.com/dvirross/leveraged-etf-compounding.git
cd leveraged-etf-compounding
python -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt
python -m analysis.run                                # tables in results/, figures in figures/
jupyter nbconvert --to notebook --execute --inplace notebooks/leveraged-etf-empirical-analysis.ipynb
pytest -q                                             # unit tests
```

No internet access is needed. The static PNG fallbacks need `kaleido` and a local Chrome or Chromium (set `BROWSER_PATH` if it is not found); without them the JSON specs are still written and a message is printed.

## Limitations

Five years is one episode with overlapping windows, so the results describe that history and are not forecasts. The idealized product ignores fees, financing, tracking error and distributions; actual TQQQ trails it by a median of about 3 percentage points per 63-day window in this sample, and the analysis does not decompose that gap. A longer history (additional years before 2021, especially the 2008 and 2000–2002 periods) would show how stable the relationships are; the loader accepts files with the same columns. This is educational mathematics, not investment advice.

## Author and license

[Dvir Ross](https://dvirross.com). Code and notebooks: MIT license. Third-party market data and the sources cited in the article are not relicensed (see [LICENSE](LICENSE) and [data/README.md](data/README.md)).
