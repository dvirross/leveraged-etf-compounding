"""Unit tests. The tiny series below are hand-made FIXTURES used only to test the code; they are not market data and
are not used for any figure or number in the article."""
import numpy as np
import pandas as pd
import pytest

from analysis import empirical as e


def _series(returns, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(returns) + 1)
    return pd.Series(100 * np.r_[1.0, np.cumprod(1 + np.asarray(returns))], index=idx)


def test_two_day_closed_form():
    x = 0.10
    s = _series([-x, x / (1 - x)])
    w = e.rolling_windows(s, 2)
    assert w.bench_return.iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert w.lev_return.iloc[0] == pytest.approx(-3 * 2 * x**2 / (1 - x), rel=1e-12)   # 1 - L(L-1)x^2/(1-x) - 1


def test_reordering_returns_does_not_change_terminal_wealth():
    r = np.array([0.03, -0.02, 0.01, -0.04, 0.05])
    a = e.rolling_windows(_series(r), 5).lev_return.iloc[0]
    b = e.rolling_windows(_series(r[::-1]), 5).lev_return.iloc[0]
    assert a == pytest.approx(b, rel=1e-12)


def test_drag_matches_second_order_approximation_for_small_moves():
    rng = np.random.default_rng(0)                          # fixture noise, not market data
    w = e.rolling_windows(_series(rng.normal(0.0004, 0.004, 63)), 63)
    approx = np.exp(-3 * w.sum_sq.iloc[0]) - 1
    assert w.drag.iloc[0] == pytest.approx(approx, abs=2e-4)


def test_match_windows_respects_rule():
    # hand-made window table (fixture): rows 0, 10, 20 qualify; row 5 overlaps row 0; row 30 has a different up-day count;
    # row 40 has an endpoint outside the tolerance band
    win = pd.DataFrame({
        "base_date": pd.bdate_range("2020-01-01", periods=50), "end_date": pd.bdate_range("2020-01-15", periods=50),
        "bench_return": 0.06, "lev_return": 0.20, "pos_days": 12, "vol": 0.2})
    win.loc[10, "lev_return"] = 0.25
    win.loc[20, "lev_return"] = 0.15
    win.loc[30, "pos_days"] = 11
    win.loc[40, "bench_return"] = 0.09
    win = win[win.index.isin([0, 5, 10, 20, 30, 40])]
    res = e.match_windows(win, horizon=5, tolerance=0.005, min_return=0.05)
    assert list(res.windows.index) == [0, 1, 2]
    assert sorted(res.windows.lev_return) == [0.15, 0.20, 0.25]


def test_round_trip_search_finds_fall_and_recovery():
    q = _series([-0.02, 0.0204])
    df = pd.DataFrame({"QQQ": q, "TQQQ": q, "SQQQ": q})
    out = e.round_trip_search(df, tolerance=0.001)
    assert len(out) == 1 and out.r1.iloc[0] < 0
