"""Tests of the exact decomposition R_L = L R + [(1+R)^L - 1 - L R] + (1+R)^L (exp(D) - 1).

The short return arrays are hand-made or fixed-seed FIXTURES used only to test the algebra. The final block (marked REGRESSION)
reads the committed five-year IBKR data and checks the numbers published in the article."""
import numpy as np
import pandas as pd
import pytest

from analysis import empirical as e
from analysis import figures as figs
from analysis import run


def _fixture_returns(n=20, seed=3, scale=0.02):
    return np.random.default_rng(seed).normal(0.003, scale, n)


def test_identity_reconstructs_the_leveraged_return_exactly():
    for seed in range(5):
        for lev in (2, 3, 4):
            d = e.decompose(_fixture_returns(seed=seed), lev)
            assert abs(d["reconstruction_error"]) < 1e-13
            assert 1 + d["lev_return"] == pytest.approx((1 + d["bench_return"]) ** lev * np.exp(d["D_exact"]), rel=1e-13)


def test_signs_each_daily_term_is_nonpositive_and_benefit_positive():
    r = _fixture_returns(60, seed=11, scale=0.03)
    d = e.decompose(r)
    contrib = np.log1p(3 * r) - 3 * np.log1p(r)
    assert (contrib <= 0).all() and d["D_exact"] < 0 and d["volatility_correction"] < 0
    assert d["compounding_benefit"] == pytest.approx(3 * d["bench_return"] ** 2 + d["bench_return"] ** 3, rel=1e-12)
    assert d["compounding_benefit"] > 0


def test_flat_market_has_no_correction_and_leverage_one_has_no_split():
    flat = e.decompose(np.zeros(5))
    assert flat["volatility_correction"] == 0 and flat["compounding_benefit"] == 0 and flat["lev_return"] == 0
    one = e.decompose(_fixture_returns(), leverage=1)
    assert one["compounding_benefit"] == pytest.approx(0, abs=1e-15) and one["volatility_correction"] == pytest.approx(0, abs=1e-15)


def test_two_day_case_matches_the_closed_form():
    a, b = 0.10, -1 / 11                                    # QQQ returns to its start: R = 0
    d = e.decompose([a, b])
    assert d["bench_return"] == pytest.approx(0, abs=1e-15)
    assert d["lev_return"] == pytest.approx(3 * 0 + 6 * a * b, rel=1e-12)          # R_3 = 3R + 6ab
    assert d["volatility_correction"] == pytest.approx(d["lev_return"], rel=1e-12)  # all of it is correction when R = 0
    assert e.decompose([b, a])["lev_return"] == pytest.approx(d["lev_return"], rel=1e-14)   # order does not matter


def test_second_order_approximation_is_accurate_for_small_moves_and_errs_for_large_ones():
    small = e.decompose(np.random.default_rng(1).normal(0, 0.003, 20))
    assert abs(small["second_order_error"]) < 1e-6
    big = e.decompose([0.12, -0.02, 0.01])
    assert abs(big["D_second_order"] - big["D_exact"]) > 1e-3                      # large daily returns: higher-order terms matter
    assert big["D_second_order"] == pytest.approx(-3 * big["sum_sq"], rel=1e-14)


def test_rejects_returns_that_wipe_out_the_product():
    with pytest.raises(ValueError):
        e.decompose([0.01, -0.34])                                                 # 1 + 3r < 0


def test_daily_contributions_sum_to_D_and_shares_to_one():
    idx = pd.bdate_range("2020-01-01", periods=21)
    r = _fixture_returns(20)
    close = pd.Series(100 * np.r_[1.0, np.cumprod(1 + r)], index=idx)
    win = pd.DataFrame({"base_date": [idx[0]], "end_date": [idx[-1]]})
    t = e.daily_contributions(close, win)
    d = e.decompose_windows(close, win).iloc[0]
    assert len(t) == 20 and t.contribution_D.sum() == pytest.approx(d.D_exact, rel=1e-12)
    assert t.share_of_D.sum() == pytest.approx(1.0) and t.share_of_sum_sq.sum() == pytest.approx(1.0)
    assert t.second_order_contribution.sum() == pytest.approx(d.D_second_order, rel=1e-12)


def test_rolling_summary_uses_arithmetic_and_relative_corrections_consistently():
    rng = np.random.default_rng(5)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.012, 300)), index=pd.bdate_range("2020-01-01", periods=300))
    w = e.rolling_windows(close, 40)
    d = e.add_decomposition(w)
    corr_direct = (1 + w.bench_return) ** 3 * w.drag                               # arithmetic correction = (1+R)^3 (e^D - 1); drag = e^D - 1
    assert np.allclose(d.volatility_correction, corr_direct, atol=1e-13)
    s = e.rolling_decomposition_summary(w).set_index("statistic").value
    assert s["windows"] == len(w) and s["relative_correction_max_pct"] <= 0
    assert s["arithmetic_correction_min_pp"] <= s["arithmetic_correction_median_pp"] <= s["arithmetic_correction_max_pp"] <= 0


def test_figure_builder_plots_the_table_values_in_percentage_points():
    dec = pd.DataFrame(dict(base_date=pd.to_datetime(["2022-06-17"]), end_date=pd.to_datetime(["2022-07-19"]), naive_return=[0.25], lev_return=[0.26], bench_return=[0.083],
                            compounding_benefit=[0.02], volatility_correction=[-0.01], net_vs_naive=[0.01], D_exact=[-0.012]))
    f = figs.fig_window_decomposition(dec)
    assert list(f.data[0].y) == [2.0] and list(f.data[1].y) == [-1.0] and list(f.data[2].y) == pytest.approx([1.0])
    assert f.data[0].x[0].startswith("Jun–Jul 2022")


# ---------------------------------------------------------------------------------------------- REGRESSION (committed real data)
@pytest.fixture(scope="module")
def real():
    close = e.load_aligned(run.DATA)
    w20 = e.rolling_windows(close["QQQ"], run.MATCH_HORIZON, compare=close["TQQQ"])
    m = e.match_windows(w20, run.MATCH_HORIZON, run.MATCH_TOLERANCE, run.MATCH_MIN_RETURN)
    w63 = e.rolling_windows(close["QQQ"], run.ROLL_HORIZON, compare=close["TQQQ"])
    return close, m, w63


def test_published_three_window_numbers(real):
    close, m, _ = real
    t = e.decompose_windows(close["QQQ"], m.windows)
    assert t.n_returns.tolist() == [20, 20, 20] and t.reconstruction_error.abs().max() < 1e-13
    pp = lambda c: (100 * t[c]).round(2).tolist()
    assert pp("naive_return") == [25.79, 26.87, 25.41]
    assert pp("compounding_benefit") == [2.28, 2.48, 2.21]
    assert pp("volatility_correction") == [-2.16, -1.00, -8.24]
    assert pp("lev_return") == [25.90, 28.35, 19.38]
    assert pp("net_vs_naive") == [0.12, 1.47, -6.03]
    gap = t.lev_return[1] - t.lev_return[2]
    assert 100 * gap == pytest.approx(8.97, abs=0.005)
    assert 100 * (t.volatility_correction[1] - t.volatility_correction[2]) == pytest.approx(7.24, abs=0.005)
    assert 100 * (gap - (t.volatility_correction[1] - t.volatility_correction[2])) == pytest.approx(1.73, abs=0.005)


def test_published_second_order_numbers(real):
    close, m, _ = real
    t = e.decompose_windows(close["QQQ"], m.windows)
    assert t.sum_sq.round(5).tolist() == [0.00583, 0.00272, 0.02487]
    assert (100 * t.D_second_order).round(2).tolist() == [-1.75, -0.82, -7.46]
    assert (100 * t.D_exact).round(2).tolist() == [-1.70, -0.78, -6.68]
    assert (100 * t.lev_return_second_order).round(2).tolist() == [25.85, 28.30, 18.45]
    assert 100 * abs(t.second_order_error[2]) == pytest.approx(0.93, abs=0.005)


def test_published_april_2025_day_contributions(real):
    close, m, _ = real
    c = e.daily_contributions(close["QQQ"], m.windows)
    c = c[c.window == 3].set_index(c[c.window == 3].date.dt.strftime("%Y-%m-%d"))
    a9, a4 = c.loc["2025-04-09"], c.loc["2025-04-04"]
    assert a9.r == pytest.approx(0.1200, abs=5e-5) and a4.r == pytest.approx(-0.0621, abs=5e-5)
    assert 100 * a9.contribution_D == pytest.approx(-3.25, abs=0.005) and a9.growth_factor_levered == pytest.approx(1.360, abs=5e-4)
    assert a9.growth_factor_cubed == pytest.approx(1.405, abs=5e-4)
    assert round(100 * a9.share_of_D, 1) == 48.7 and round(100 * a4.share_of_D, 1) == 20.7 and round(100 * a9.share_of_sum_sq, 1) == 57.9
    assert round(100 * (a9.share_of_D + a4.share_of_D), 1) == 69.4
    assert 100 * a9.second_order_contribution == pytest.approx(-4.32, abs=0.005) and 100 * a4.second_order_contribution == pytest.approx(-1.16, abs=0.005)
    assert 100 * a4.contribution_D == pytest.approx(-1.38, abs=0.005)


def test_published_63_day_summary(real):
    _, _, w63 = real
    s = e.rolling_decomposition_summary(w63).set_index("statistic").value
    assert s["windows"] == 1190 and s["windows_bench_up"] == 796
    assert round(s["relative_correction_min_pct"], 2) == -10.55 and round(s["relative_correction_max_pct"], 2) == -0.85
    assert round(s["arithmetic_correction_min_pp"], 2) == -15.03 and round(s["arithmetic_correction_max_pp"], 2) == -1.11
    assert round(s["arithmetic_correction_median_pp"], 1) == -3.3 and round(s["compounding_benefit_median_pp"], 1) == 1.9
    assert s["up_windows_benefit_exceeds_abs_correction"] == 325 == s["up_windows_lev_beats_naive"]


def test_committed_result_tables_match_a_fresh_computation(real):
    close, m, w63 = real
    for name, fresh in run.decomposition_tables(close, m.windows, w63).items():
        saved = pd.read_csv(run.RESULTS / name, parse_dates=[c for c in ("base_date", "end_date", "date") if c in fresh.columns])
        pd.testing.assert_frame_equal(saved, fresh, check_dtype=False, rtol=1e-12, atol=1e-14)
