# Why a daily 3× leveraged product is not 3× the long-term return

Reproducible analysis behind the article **[Why 3× Daily Leverage Is Not 3× Long-Term Return](https://dvirross.com/quant-research/notes/why-3x-daily-leverage-is-not-3x-long-term-return)** on dvirross.com. The article explains the ideas for a general reader; this repository holds the code, the data snapshot and the figures so that every number can be checked.

## The question

A daily 3× leveraged ETF targets three times the index's return **each day**. Compounded over many days, the result is not three times the index's cumulative return. How large is the gap, what drives it, and where does the simple model stop matching real funds?

## The mechanism in one example

An index falls 10% and then rises by exactly the 11.1% needed to recover. It is back at its start. An idealized daily 3× product falls 30%, rises 33.3% and ends 6.7% below its start. In general the shortfall after a fall of *x* and an exact recovery is `L(L−1)x² / (1−x)`, which grows with the **square** of the move. Compounding can also work *for* a leveraged product in a steady trend, so this is a statement about path dependence, not "leveraged ETFs always decay".

## What the notebook does

[`notebooks/leveraged-etf-volatility-drag.ipynb`](notebooks/leveraged-etf-volatility-drag.ipynb) runs top to bottom in about a minute with fixed random seeds:

1. a closed-form two-day example;
2. daily compounding versus "multiply the cumulative return by L", with a check of the second-order approximation;
3. path dependence: three index paths with the same +10% return and different 3× outcomes;
4. a **synthetic** volatility experiment (arithmetic drift fixed at 8% a year, volatility 10–60%, five years), separating arithmetic mean returns from compound growth;
5. how approximate long-run growth depends on the leverage level, `g(L) ≈ Lμ − ½L²σ²`;
6. a vectorized Monte Carlo of terminal wealth (100,000 paths) with sensitivity to drift and volatility and a fat-tailed (Student-t) check, reporting medians, means and the probability of finishing below the start;
7. a **historical** illustration: QQQ, TQQQ and SQQQ over 21–25 June 2024.

## Synthetic versus historical

Sections 1–6 are mathematics and simulation under stated assumptions (i.i.d. returns, constant drift and volatility, no fees, financing or tracking error). They are **not** forecasts and not calibrated to any market. Section 7 is the only real market data: three closing prices per fund.

## The June 2024 example

| 21 → 25 June 2024 | Cumulative return |
|---|---:|
| QQQ (dividend-adjusted) | −0.008% |
| TQQQ | −0.161% |
| SQQQ | −0.119% |

QQQ is essentially flat **on a dividend-adjusted basis** (its raw price return was about −0.17% because it went ex-dividend on 24 June), while both the +3× and the −3× product finished slightly below their starting values. Qualifications that matter:

- The three-date snapshot was **recovered from the author's original Colab output**, not freshly downloaded. It uses `yfinance` auto-adjusted `Close`, and the original retrieval date was not preserved. See [`data/README.md`](data/README.md).
- QQQ is a tradable proxy for the Nasdaq-100; TQQQ and SQQQ target the **index**, not QQQ shares.
- The example was **selected**: the original notebook searched for a loss-and-rebound sequence. It illustrates a mechanism; it is not evidence about how often this happens or about expected returns.
- Observed fund returns also include fees, financing, tracking and timing effects, so the deviation from "3× the index" is not attributed entirely to compounding.

## Reproduce it

```bash
git clone https://github.com/dvirross/leveraged-etf-compounding.git
cd leveraged-etf-compounding
python -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/leveraged-etf-volatility-drag.ipynb
```

No internet access is needed; the real-data section reads the committed CSV. Figures are written to `figures/` together with a `figures.json` manifest (captions and assumptions). An optional cell compares the snapshot's returns with a fresh Yahoo Finance download if `yfinance` is installed and reachable; the run never depends on it.

## Contents

```
notebooks/   the executed notebook
data/        three-date CSV snapshot and its provenance notes
figures/     publication figures exported by the notebook
requirements.txt, LICENSE
```

## Methodological caveats

Idealized daily reset only; i.i.d. normal returns (with one Student-t check); constant drift and volatility; no volatility clustering; second-order growth approximation; one selected historical example. Terminal wealth is highly skewed, so means, medians and downside probabilities tell different stories. This is educational mathematics, not investment advice or a trading recommendation.

## Author and license

[Dvir Ross](https://dvirross.com). Code and notebook: MIT license. Third-party market data and the sources cited in the article are not relicensed (see [LICENSE](LICENSE) and [data/README.md](data/README.md)).
