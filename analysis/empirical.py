"""Rolling-window analysis of an idealized daily-reset leveraged product on observed price history.

Everything here works on simple close-to-close returns of the observed series. The "idealized" L-times product is the
rule  value_t = value_{t-1} * (1 + L * r_t)  applied to the benchmark's own daily returns. It ignores fees, financing,
tracking error and distributions, so it is a model of the daily-reset mechanism, not a replica of any fund.

Conventions
-----------
* A window of horizon H is H consecutive daily returns. Its *base date* is the close before the first return and its
  *end date* is the close of the last return, so a window covers H+1 closes.
* Realized volatility is the sample standard deviation (ddof=1) of the H daily returns times sqrt(252).
* ``drag`` is ``(1 + L-times return) / (1 + R)^L - 1``, where R is the benchmark cumulative return over the window. ``(1 + R)^L`` is the
  benchmark's wealth raised to the leverage power, a path-independent reference (the benchmark's cumulative log return multiplied by L and
  exponentiated). The ratio removes that cumulative log-return term. To second order, log(1 + L-times return) - L log(1 + R) is
  approximately -L(L-1)/2 * sum(r^2); higher-order terms remain, so the ratio is not exactly a pure volatility measure.
* ``deviation`` is idealized leveraged return minus L times the benchmark cumulative return (the "naive" multiple).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
LEVERAGE = 3


def load_close(path: str | Path) -> pd.Series:
    """Read an IBKR daily-bar CSV (columns date, open, high, low, close, volume) and return the close series."""
    df = pd.read_csv(path, parse_dates=["date"])
    s = df.set_index("date")["close"].astype(float)
    if not s.index.is_monotonic_increasing or s.index.has_duplicates:
        raise ValueError(f"{path}: dates must be strictly increasing")
    if s.isna().any() or (s <= 0).any():
        raise ValueError(f"{path}: missing or non-positive closes")
    return s


def load_aligned(data_dir: str | Path) -> pd.DataFrame:
    """QQQ, TQQQ and SQQQ closes on a common set of dates (the files are expected to coincide exactly)."""
    data_dir = Path(data_dir)
    out = pd.DataFrame({t.upper(): load_close(data_dir / f"ibkr-{t}-daily-5y.csv") for t in ("qqq", "tqqq", "sqqq")})
    if out.isna().any().any():
        raise ValueError("QQQ, TQQQ and SQQQ files do not cover the same dates")
    return out


def rolling_windows(close: pd.Series, horizon: int, leverage: int = LEVERAGE, compare: pd.Series | None = None) -> pd.DataFrame:
    """All overlapping windows of ``horizon`` daily returns, one row per window.

    ``compare`` is an optional second close series on the same dates (for example the actual TQQQ); its window
    return is reported as ``actual_return``. Overlapping windows are strongly dependent; they are not independent samples.
    """
    r = close.pct_change().dropna()
    rv = r.values
    n = len(rv) - horizon + 1
    if n <= 0:
        raise ValueError("series shorter than the horizon")
    rows = []
    cr = compare.pct_change().dropna().values if compare is not None else None
    for i in range(n):
        w = rv[i:i + horizon]
        path = np.cumprod(1 + w)
        lev = np.cumprod(1 + leverage * w)
        peak = np.maximum.accumulate(np.r_[1.0, path])[1:]
        R = path[-1] - 1
        L = lev[-1] - 1
        row = dict(
            base_date=close.index[i], end_date=close.index[i + horizon],
            bench_return=R, lev_return=L, naive_return=leverage * R,
            deviation=L - leverage * R,
            drag=(1 + L) / (1 + R) ** leverage - 1,
            vol=w.std(ddof=1) * np.sqrt(TRADING_DAYS),
            pos_days=int((w > 0).sum()), pos_share=float((w > 0).mean()),
            bench_max_drawdown=float((path / peak - 1).min()),
            sum_sq=float((w ** 2).sum()),
        )
        if cr is not None:
            row["actual_return"] = float(np.prod(1 + cr[i:i + horizon]) - 1)
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class MatchResult:
    horizon: int
    tolerance: float
    min_return: float
    n_candidates: int
    windows: pd.DataFrame       # the selected windows, chronological
    runners_up: pd.DataFrame    # next-best triples (rank, rows) for transparency


def match_windows(win: pd.DataFrame, horizon: int, tolerance: float = 0.005, min_return: float = 0.05, k: int = 3,
                  top: int = 5) -> MatchResult:
    """Systematic search for ``k`` non-overlapping windows that share a benchmark outcome but not a leveraged one.

    Rule (fixed before looking at any result):
      1. consider all rolling windows of the given horizon with benchmark cumulative return >= ``min_return``;
      2. keep sets of ``k`` windows that do not overlap in time, have the *same number of positive-return days*, and
         whose benchmark cumulative returns all lie within a band of width ``tolerance`` (default 0.5 percentage points);
      3. among all such sets, report the one with the largest range of idealized leveraged return; if several sets share
         the same extreme windows, prefer the one whose smallest gap between consecutive leveraged outcomes is largest
         (so that the three outcomes are well separated).
    The selection is made to illustrate a mechanism, so step 3 deliberately looks for a visible spread; it is not a random draw.
    """
    w = win[win.bench_return >= min_return].reset_index()          # keep original index in column 'index'
    cands = []
    for _, grp in w.groupby("pos_days"):
        g = grp.sort_values("index")
        idx = g["index"].values
        R = g.bench_return.values
        pos = np.arange(len(g))
        for combo in combinations(pos, k):
            ii = idx[list(combo)]
            if np.any(np.diff(ii) < horizon):                      # windows must not overlap
                continue
            rr = R[list(combo)]
            if rr.max() - rr.min() > tolerance:
                continue
            ll = np.sort(g.lev_return.values[list(combo)])
            cands.append((round(ll[-1] - ll[0], 12), float(np.diff(ll).min()), tuple(ii)))
    cands.sort(key=lambda t: (-t[0], -t[1]))
    if not cands:
        raise ValueError("no set of windows satisfies the matching rule")
    best = win.loc[list(cands[0][2])].copy()
    runners = []
    for rank, (spread, gap, ii) in enumerate(cands[:top], start=1):
        part = win.loc[list(ii), ["base_date", "end_date", "bench_return", "lev_return", "vol", "pos_days"]].copy()
        part.insert(0, "rank", rank)
        part.insert(1, "lev_range", spread)
        part.insert(2, "min_gap", gap)
        runners.append(part)
    return MatchResult(horizon, tolerance, min_return, len(cands), best.reset_index(drop=True), pd.concat(runners, ignore_index=True))


def window_path(close: pd.Series, base_date, end_date, leverage: int = LEVERAGE) -> pd.DataFrame:
    """Daily path inside one window, normalised to 100 at the base date, for plotting."""
    c = close.loc[base_date:end_date]
    r = c.pct_change().fillna(0.0)
    out = pd.DataFrame({"date": c.index, "bench": 100 * (1 + r).cumprod().values, "lev": 100 * (1 + leverage * r).cumprod().values,
                        "daily_return": r.values})
    out["day"] = np.arange(len(out))
    out["vol_to_date"] = np.nan
    return out


def round_trip_search(close: pd.DataFrame, tolerance: float = 0.001, first_day_fall: bool = True) -> pd.DataFrame:
    """Two-day windows (three closes) in which QQQ returns to within ``tolerance`` of its start.

    Rule: keep windows whose two-day QQQ return is within +-tolerance (default 0.10%), optionally require the first
    day to be a fall (a fall-and-recovery), and rank by the size of the first-day move.
    """
    rows = []
    c = close.values
    for i in range(len(close) - 2):
        r1 = c[i + 1, 0] / c[i, 0] - 1
        r2 = c[i + 2, 0] / c[i + 1, 0] - 1
        R = c[i + 2, 0] / c[i, 0] - 1
        if abs(R) > tolerance or (first_day_fall and r1 >= 0):
            continue
        rows.append(dict(base_date=close.index[i], mid_date=close.index[i + 1], end_date=close.index[i + 2], r1=r1, r2=r2, qqq_return=R,
                         ideal_plus3=(1 + 3 * r1) * (1 + 3 * r2) - 1, ideal_minus3=(1 - 3 * r1) * (1 - 3 * r2) - 1,
                         tqqq_return=c[i + 2, 1] / c[i, 1] - 1, sqqq_return=c[i + 2, 2] / c[i, 2] - 1))
    return pd.DataFrame(rows).sort_values("r1").reset_index(drop=True)


def bin_by_positive_share(win: pd.DataFrame, edges=(0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0)) -> pd.DataFrame:
    """Summary of idealized leveraged return by share of positive-return days."""
    b = pd.cut(win.pos_share, list(edges), include_lowest=True)
    return win.groupby(b, observed=True).agg(
        n=("lev_return", "size"), share_positive=("lev_return", lambda x: float((x > 0).mean())), bench_median=("bench_return", "median"), lev_median=("lev_return", "median"),
        lev_min=("lev_return", "min"), lev_max=("lev_return", "max")).reset_index().rename(columns={"pos_share": "positive_day_share"})


# ----------------------------------------------------------------------------------------------------------------------
# Exact decomposition of the idealized L-times return (article section "Explaining the three 20-day windows")
# ----------------------------------------------------------------------------------------------------------------------
# For daily returns r_t with 1 + r_t > 0 and 1 + L r_t > 0, let R be the benchmark cumulative return, R_L the idealized
# L-times cumulative return and
#     D = sum_t [ log(1 + L r_t) - L log(1 + r_t) ]          (exact log deviation; every term is <= 0)
# Then 1 + R_L = (1 + R)^L exp(D) exactly, and subtracting 1 gives, with (1 + R)^L expanded around 1 + L R,
#     R_L = L R  +  [ (1 + R)^L - 1 - L R ]  +  (1 + R)^L (exp(D) - 1)
#           naive      compounding benefit       volatility correction
# (for L = 3 the benefit is 3 R^2 + R^3). The split is an explanatory accounting identity for the idealized product, not
# the unique decomposition of leveraged-ETF returns, and real funds add fees, financing and tracking effects.
# D_2 = -L (L - 1) / 2 * sum_t r_t^2 is only the second-order Taylor approximation of D.

def window_returns(close: pd.Series, base_date, end_date) -> pd.Series:
    """The daily returns of one window: closes from ``base_date`` to ``end_date`` give ``len(close.loc[base:end]) - 1`` returns."""
    r = close.loc[base_date:end_date].pct_change().dropna()
    if r.empty:
        raise ValueError("window contains no returns")
    return r


def decompose(returns, leverage: int = LEVERAGE) -> dict:
    """Exact decomposition of one window of daily returns (full precision). See the module section above for the identities."""
    r = np.asarray(returns, dtype=float)
    if (1 + r <= 0).any() or (1 + leverage * r <= 0).any():
        raise ValueError("the decomposition needs 1 + r > 0 and 1 + L r > 0 on every day")
    R = float(np.prod(1 + r) - 1)
    R_L = float(np.prod(1 + leverage * r) - 1)
    contrib = np.log1p(leverage * r) - leverage * np.log1p(r)
    D = float(contrib.sum())
    s2 = float(np.sum(r ** 2))
    benefit = (1 + R) ** leverage - 1 - leverage * R
    correction = (1 + R) ** leverage * (np.exp(D) - 1)
    D2 = -leverage * (leverage - 1) / 2 * s2
    R_L_2 = (1 + R) ** leverage * np.exp(D2) - 1
    return dict(n_returns=len(r), bench_return=R, naive_return=leverage * R, compounding_benefit=float(benefit), volatility_correction=float(correction),
                lev_return=R_L, net_vs_naive=R_L - leverage * R, reconstruction_error=float(leverage * R + benefit + correction - R_L),
                D_exact=D, sum_sq=s2, D_second_order=D2, lev_return_second_order=float(R_L_2), second_order_error=float(R_L_2 - R_L))


def decompose_windows(close: pd.Series, windows: pd.DataFrame, leverage: int = LEVERAGE) -> pd.DataFrame:
    """One row per selected window (columns base_date, end_date as in ``match_windows``), with the exact decomposition."""
    rows = []
    for i, (_, w) in enumerate(windows.iterrows(), start=1):
        rows.append(dict(window=i, base_date=w.base_date, end_date=w.end_date, **decompose(window_returns(close, w.base_date, w.end_date), leverage)))
    return pd.DataFrame(rows)


def daily_contributions(close: pd.Series, windows: pd.DataFrame, leverage: int = LEVERAGE) -> pd.DataFrame:
    """Each day's exact contribution to D, its share of D, and the second-order contribution -L(L-1)/2 r^2 and its share of sum r^2."""
    rows = []
    for i, (_, w) in enumerate(windows.iterrows(), start=1):
        r = window_returns(close, w.base_date, w.end_date)
        c = np.log1p(leverage * r) - leverage * np.log1p(r)
        s2 = float((r ** 2).sum())
        for d, ri, ci in zip(r.index, r.values, c.values):
            rows.append(dict(window=i, date=d, r=ri, growth_factor_levered=1 + leverage * ri, growth_factor_cubed=(1 + ri) ** leverage, contribution_D=ci,
                             share_of_D=ci / c.sum(), second_order_contribution=-leverage * (leverage - 1) / 2 * ri ** 2, share_of_sum_sq=ri ** 2 / s2))
    return pd.DataFrame(rows)


def add_decomposition(win: pd.DataFrame, leverage: int = LEVERAGE) -> pd.DataFrame:
    """Rolling-window table plus the arithmetic decomposition of each window.

    ``drag`` (already in the table) is the *relative* correction exp(D) - 1 = (1 + R_L) / (1 + R)^L - 1;
    ``volatility_correction`` is the *arithmetic* correction (1 + R)^L (exp(D) - 1) in return units, which equals ``deviation - compounding_benefit``.
    """
    out = win.copy()
    out["compounding_benefit"] = (1 + out.bench_return) ** leverage - 1 - leverage * out.bench_return
    out["volatility_correction"] = out.deviation - out.compounding_benefit
    return out


def rolling_decomposition_summary(win: pd.DataFrame, leverage: int = LEVERAGE) -> pd.DataFrame:
    """Summary (one row per statistic) of the decomposition over all rolling windows; ``pp`` means percentage points of return."""
    d = add_decomposition(win, leverage)
    up = d[d.bench_return > 0]
    stats = {
        "windows": len(d),
        "relative_correction_min_pct": 100 * d.drag.min(), "relative_correction_median_pct": 100 * d.drag.median(), "relative_correction_max_pct": 100 * d.drag.max(),
        "arithmetic_correction_min_pp": 100 * d.volatility_correction.min(), "arithmetic_correction_median_pp": 100 * d.volatility_correction.median(),
        "arithmetic_correction_max_pp": 100 * d.volatility_correction.max(),
        "compounding_benefit_median_pp": 100 * d.compounding_benefit.median(),
        "windows_bench_up": len(up),
        "up_windows_benefit_exceeds_abs_correction": int((up.compounding_benefit > -up.volatility_correction).sum()),
        "up_windows_lev_beats_naive": int((up.deviation > 0).sum()),
    }
    return pd.DataFrame({"statistic": list(stats), "value": list(stats.values())})
