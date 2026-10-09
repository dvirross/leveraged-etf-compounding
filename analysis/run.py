"""Regenerate every derived table and figure from the committed five-year IBKR data.

    python -m analysis.run            # from the repository root

Writes full-precision CSVs to results/ and Plotly specs (+ PNG fallbacks when possible) to figures/.
Article figure numbers: files 02, 03, 03b, 04, 05 are article Figures 2, 3, 4, 5, 6 (03b is the decomposition added after the original series).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import empirical as emp
from . import figures as figs

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS, FIGS = ROOT / "data", ROOT / "results", ROOT / "figures"

MATCH_HORIZON = 20          # about one trading month
ROLL_HORIZON = 63           # about one trading quarter (see results/horizon_sensitivity.csv)
MATCH_TOLERANCE = 0.005     # benchmark returns of matched windows lie within 0.5 percentage points
MATCH_MIN_RETURN = 0.05


def horizon_table(close: pd.DataFrame, horizons=(20, 63, 126, 252)) -> pd.DataFrame:
    rows = []
    for h in horizons:
        w = emp.rolling_windows(close["QQQ"], h, compare=close["TQQQ"])
        rows.append(dict(
            horizon_days=h, overlapping_windows=len(w), approx_independent_windows=(len(close) - 1) // h,
            share_lev_below_zero=(w.lev_return < 0).mean(), windows_bench_up_lev_down=int(((w.bench_return > 0) & (w.lev_return < 0)).sum()),
            share_lev_above_naive=(w.deviation > 0).mean(), median_drag=w.drag.median(), corr_vol_drag=w[["vol", "drag"]].corr().iloc[0, 1],
            spearman_posshare_levreturn=w[["pos_share", "lev_return"]].corr("spearman").iloc[0, 1],
            rms_actual_tqqq_minus_ideal=float(np.sqrt(((w.actual_return - w.lev_return) ** 2).mean()))))
    return pd.DataFrame(rows)


def decomposition_tables(close: pd.DataFrame, windows: pd.DataFrame, rolling: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Result tables of the exact compounding-versus-volatility-correction decomposition (file name -> DataFrame).

    ``windows`` are the three matched 20-day windows and ``rolling`` the 63-day rolling windows. Used by ``main`` and by the notebook,
    so there is a single implementation of what is written to ``results/``.
    """
    return {
        "window_decomposition_20d.csv": emp.decompose_windows(close["QQQ"], windows),
        "window_daily_contributions_20d.csv": emp.daily_contributions(close["QQQ"], windows),
        f"rolling_{ROLL_HORIZON}d_decomposition_summary.csv": emp.rolling_decomposition_summary(rolling),
    }


def main(png: bool = True) -> None:
    RESULTS.mkdir(exist_ok=True)
    close = emp.load_aligned(DATA)

    # Figure 2: real two-session fall and recovery
    rt = emp.round_trip_search(close)
    rt.to_csv(RESULTS / "round_trip_candidates.csv", index=False)
    figs.export(figs.fig_round_trip(close, rt.iloc[0]), FIGS, "02-real-two-day-roundtrip", (1100, 520), png)

    # Figure 3: matched windows
    w20 = emp.rolling_windows(close["QQQ"], MATCH_HORIZON, compare=close["TQQQ"])
    m = emp.match_windows(w20, MATCH_HORIZON, MATCH_TOLERANCE, MATCH_MIN_RETURN)
    m.windows.to_csv(RESULTS / "matched_windows_20d.csv", index=False)
    m.runners_up.to_csv(RESULTS / "matched_windows_20d_top_candidates.csv", index=False)
    paths = pd.concat([emp.window_path(close["QQQ"], w.base_date, w.end_date).assign(window=i + 1) for i, (_, w) in enumerate(m.windows.iterrows())])
    paths.drop(columns="vol_to_date").to_csv(RESULTS / "matched_windows_20d_daily_paths.csv", index=False)
    figs.export(figs.fig_matched_windows(close["QQQ"], m.windows, MATCH_HORIZON), FIGS, "03-matched-windows", (1100, 760), png)

    # Figures 5 and 6 (files 04 and 05): rolling windows
    w = emp.rolling_windows(close["QQQ"], ROLL_HORIZON, compare=close["TQQQ"])
    w.to_csv(RESULTS / f"rolling_{ROLL_HORIZON}d_windows.csv", index=False)
    emp.bin_by_positive_share(w).to_csv(RESULTS / f"positive_day_bins_{ROLL_HORIZON}d.csv", index=False)
    horizon_table(close).to_csv(RESULTS / "horizon_sensitivity.csv", index=False)
    figs.export(figs.fig_vol_drag(w, ROLL_HORIZON), FIGS, "04-volatility-drag-rolling", (1100, 640), png)
    figs.export(figs.fig_green_days(w, ROLL_HORIZON), FIGS, "05-positive-days-rolling", (1100, 560), png)

    # Article Figure 4 (file 03b): exact decomposition of the three matched windows, and its 63-day link
    tables = decomposition_tables(close, m.windows, w)
    for name, df in tables.items():
        df.to_csv(RESULTS / name, index=False)
    figs.export(figs.fig_window_decomposition(tables["window_decomposition_20d.csv"]), FIGS, "03b-window-decomposition", (1100, 640), png)

    # One manifest entry per figure is written by the notebook; this script writes the machine-readable run summary.
    summary = dict(
        data_first=str(close.index.min().date()), data_last=str(close.index.max().date()), rows=len(close),
        match=dict(horizon=MATCH_HORIZON, tolerance=MATCH_TOLERANCE, min_return=MATCH_MIN_RETURN, candidate_sets=m.n_candidates),
        rolling=dict(horizon=ROLL_HORIZON, windows=len(w)))
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
