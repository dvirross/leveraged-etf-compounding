"""Plotly figure builders for the article's real-data figures (article Figures 2 to 6).

Each builder takes the DataFrames produced by ``analysis.empirical`` and returns a ``plotly.graph_objects.Figure``.
``export`` writes the JSON spec that the website renders with plotly.js plus a static PNG fallback (PNG needs kaleido
and a local Chrome/Chromium; the JSON is the primary artifact). Percentages are displayed with two decimals; the CSV
files in ``results/`` keep full precision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from . import empirical as emp

BLUE, ORANGE, VIOLET, TEAL = "#3758d7", "#c96f33", "#7a4fb3", "#1f8a8a"
INK, MUTED, GRID, AXIS = "#142033", "#5d6678", "#e3e6ee", "#b9bfcc"
FONT = dict(family="Inter, system-ui, -apple-system, 'Segoe UI', sans-serif", size=13, color=INK)
BASE = dict(template="none", paper_bgcolor="white", plot_bgcolor="white", font=FONT)


def _d(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts:%b} {ts.day}, {ts.year}"


def _pct(x, nd=2) -> np.ndarray:
    return np.round(100 * np.asarray(x, dtype=float), nd)


# --------------------------------------------------------------------------------------- Figure 2
def fig_round_trip(close: pd.DataFrame, row: pd.Series) -> go.Figure:
    c = close.loc[row.base_date:row.end_date]
    norm = 100 * c / c.iloc[0]
    q = c["QQQ"].pct_change().fillna(0.0)
    ideal = 100 * (1 + 3 * q).cumprod()
    labels = [_d(d) for d in c.index]
    ret = 100 * (c / c.iloc[0] - 1)
    spec = [("QQQ", "QQQ (index proxy)", BLUE, "solid", norm["QQQ"], c["QQQ"]),
            ("TQQQ", "TQQQ actual (daily +3×)", ORANGE, "solid", norm["TQQQ"], c["TQQQ"]),
            ("SQQQ", "SQQQ actual (daily −3×)", VIOLET, "solid", norm["SQQQ"], c["SQQQ"])]
    fig = go.Figure()
    for key, name, col, dash, y, raw in spec:
        fig.add_trace(go.Scatter(
            x=labels, y=np.round(y.values, 6), mode="lines+markers", name=name, line=dict(color=col, width=3, dash=dash), marker=dict(size=9),
            customdata=np.column_stack([np.round(raw.values, 2), np.round(ret[key].values, 4)]),
            # absolute TQQQ/SQQQ levels may reflect retrospective split normalization in the source series, so only QQQ's close is shown
            hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>" + ("Close: %{customdata[0]:.2f}<br>" if key == "QQQ" else "") + "Value: %{y:.2f}<br>Return since " + _d(c.index[0]) + ": %{customdata[1]:+.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=labels, y=np.round(ideal.values, 6), mode="lines+markers", name="Idealized +3× applied to QQQ", line=dict(color=ORANGE, width=2, dash="dot"),
        marker=dict(size=7, symbol="diamond-open"),
        hovertemplate="<b>Idealized +3× (model)</b><br>%{x}<br>Value: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=100, line=dict(color=AXIS, width=1))
    lo, hi = float(min(norm.min().min(), ideal.min())), float(max(norm.max().max(), ideal.max()))
    pad = 0.12 * (hi - lo)
    fig.update_layout(**BASE, height=440, margin=dict(l=64, r=24, t=24, b=150), hovermode="closest",
                      legend=dict(orientation="h", x=0, y=-0.16, xanchor="left", yanchor="top", font=dict(size=12)),
                      xaxis=dict(type="category", showgrid=False, linecolor=AXIS),
                      yaxis=dict(title=f"Value ({pd.Timestamp(c.index[0]):%b} {pd.Timestamp(c.index[0]).day} = 100)", range=[lo - pad, hi + pad], tickformat=".0f", gridcolor=GRID, zeroline=False))
    return fig


# --------------------------------------------------------------------------------------- Figure 3
def fig_matched_windows(close: pd.Series, windows: pd.DataFrame, horizon: int) -> go.Figure:
    colors = [BLUE, ORANGE, TEAL]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("QQQ price (base = 100)", "Idealized daily 3× product (base = 100)"))
    naive = 100 * (1 + 3 * windows.bench_return.mean())
    for k, (_, w) in enumerate(windows.iterrows()):
        p = emp.window_path(close, w.base_date, w.end_date)
        name = f"{_d(w.base_date)} to {_d(w.end_date)}"
        dates = [_d(d) for d in p.date]
        for r, col in ((1, "bench"), (2, "lev")):
            ret = 100 * (p[col] / 100 - 1)
            fig.add_trace(go.Scatter(
                x=p.day, y=np.round(p[col].values, 4), mode="lines+markers", name=name, legendgroup=name, showlegend=(r == 1),
                line=dict(color=colors[k], width=2.6), marker=dict(size=5),
                customdata=np.column_stack([dates, np.round(ret.values, 4), np.round(100 * p.daily_return.values, 4)]),
                hovertemplate=("<b>" + name + "</b><br>Day %{x}: %{customdata[0]}<br>"
                               + ("QQQ value: " if r == 1 else "Idealized 3× value: ") + "%{y:.2f}<br>Return since base: %{customdata[1]:+.2f}%<br>"
                               "Day's QQQ return: %{customdata[2]:+.2f}%<br>"
                               f"Window realized volatility: {100 * w.vol:.2f}%<extra></extra>")), row=r, col=1)
    fig.add_hline(y=naive, row=2, col=1, line=dict(color=MUTED, width=1.4, dash="dash"),
                  annotation_text=f"3 × the QQQ gain: about {naive:.0f}", annotation_position="top left", annotation_font=dict(size=12, color=MUTED))
    fig.update_layout(**BASE, height=640, margin=dict(l=56, r=16, t=36, b=150), hovermode="closest",
                      legend=dict(orientation="h", x=0, y=-0.17, yanchor="top", xanchor="left", font=dict(size=12)))
    for a in fig.layout.annotations:
        if a.text and not a.text.startswith("3 ×"):
            a.update(x=0, xanchor="left", font=dict(size=13, color=MUTED))
    fig.update_xaxes(showgrid=False, linecolor=AXIS, dtick=5, range=[-0.4, horizon + 0.4])
    fig.update_xaxes(title_text="Trading days since the base date", row=2, col=1)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickformat=".0f")
    return fig


# --------------------------------------------------------------------------------------- Figure 4 (file 03b)
def _short_window(base, end) -> str:
    """'May 2023' when the window starts and ends in the same month, else 'Jun–Jul 2022' (windows here never span a year end)."""
    b, e = pd.Timestamp(base), pd.Timestamp(end)
    return f"{b:%b %Y}" if (b.year, b.month) == (e.year, e.month) else f"{b:%b}–{e:%b %Y}"


def fig_window_decomposition(dec: pd.DataFrame) -> go.Figure:
    """Grouped bars per window: compounding benefit (solid), volatility correction (striped) and their net, in percentage points of
    return. ``dec`` is the table from ``analysis.empirical.decompose_windows``. Window colours follow article Figure 3."""
    colors = [BLUE, ORANGE, TEAL]
    pp = lambda v: f"{100 * v:+.2f}".replace("-", "−")
    rows = [dec.iloc[i] for i in range(len(dec))]
    xl = [f"{_short_window(r.base_date, r.end_date)}<br>{100 * r.naive_return:.2f}→{100 * r.lev_return:.2f}%" for r in rows]
    full = [f"{_d(r.base_date).rsplit(',', 1)[0]} – {_d(r.end_date)}" for r in rows]
    col = colors[:len(rows)]
    txt = dict(textposition="outside", constraintext="none", textfont=dict(size=11), cliponaxis=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xl, y=[100 * r.compounding_benefit for r in rows], marker=dict(color=col), name="Compounding benefit", text=[pp(r.compounding_benefit) for r in rows],
                         offsetgroup="a", showlegend=False, customdata=[[n, 100 * r.bench_return] for n, r in zip(full, rows)], **txt,
                         hovertemplate="%{customdata[0]}<br>Compounding benefit: %{y:+.2f} pp<br>(depends only on QQQ's window return, %{customdata[1]:.2f}%)<extra></extra>"))
    fig.add_trace(go.Bar(x=xl, y=[100 * r.volatility_correction for r in rows], marker=dict(color=col, pattern=dict(shape="/", fgcolor="white", size=7, solidity=0.45)),
                         name="Volatility correction", text=[pp(r.volatility_correction) for r in rows], offsetgroup="b", showlegend=False,
                         customdata=[[n, 100 * r.D_exact] for n, r in zip(full, rows)], **txt,
                         hovertemplate="%{customdata[0]}<br>Volatility correction: %{y:+.2f} pp<br>(log deviation D = %{customdata[1]:.2f}%)<extra></extra>"))
    fig.add_trace(go.Bar(x=xl, y=[100 * r.net_vs_naive for r in rows], marker=dict(color=INK), name="Net", text=[pp(r.net_vs_naive) for r in rows], offsetgroup="c", showlegend=False,
                         customdata=[[n] for n in full], **txt, hovertemplate="%{customdata[0]}<br>Net (idealized 3× minus naive 3×): %{y:+.2f} pp<extra></extra>"))
    # legend swatches in neutral grey so the legend explains the encoding without implying a window
    fig.add_trace(go.Bar(x=[None], y=[None], name="Compounding benefit", marker=dict(color="#8a93a6")))
    fig.add_trace(go.Bar(x=[None], y=[None], name="Volatility correction", marker=dict(color="white", line=dict(color="#8a93a6", width=1),
                                                                                      pattern=dict(shape="/", fgcolor="#8a93a6", bgcolor="white", size=7, solidity=0.45))))
    fig.add_trace(go.Bar(x=[None], y=[None], name="Net versus naive 3×", marker=dict(color=INK)))
    fig.update_layout(**BASE, barmode="group", bargap=0.28, bargroupgap=0.04, height=520, margin=dict(l=62, r=12, t=22, b=120),
                      legend=dict(orientation="h", x=0, y=-0.27, yanchor="top", xanchor="left", font=dict(size=12)),
                      xaxis=dict(showgrid=False, linecolor=AXIS, tickfont=dict(size=11), tickangle=0, automargin=True),
                      yaxis=dict(title=dict(text="Percentage points"), ticksuffix=" pp", gridcolor=GRID, zeroline=True, zerolinecolor=INK, zerolinewidth=1.2, range=[-10.2, 4.4], dtick=2))
    return fig


# --------------------------------------------------------------------------------------- Figure 5
def fig_vol_drag(win: pd.DataFrame, horizon: int) -> go.Figure:
    T = horizon / emp.TRADING_DAYS
    cd = np.column_stack([[_d(d) for d in win.base_date], [_d(d) for d in win.end_date], _pct(win.bench_return), _pct(win.lev_return),
                          _pct(win.naive_return), _pct(win.deviation), _pct(win.pos_share, 1), _pct(win.vol)])
    ht = ("<b>%{customdata[0]} to %{customdata[1]}</b><br>QQQ return: %{customdata[2]:+.2f}%<br>Idealized 3× return: %{customdata[3]:+.2f}%<br>"
          "3 × QQQ return: %{customdata[4]:+.2f}%<br>Realized volatility: %{customdata[7]:.2f}%<br>Positive days: %{customdata[6]:.1f}%<br>"
          "Gap vs (1 + QQQ return)³: %{y:.2f}%<extra></extra>")
    lim = float(np.ceil(100 * win.bench_return.abs().max() / 10) * 10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_pct(win.vol), y=_pct(win.drag), mode="markers", name="Idealized 3× windows", customdata=cd, hovertemplate=ht,
        marker=dict(size=6, opacity=0.75, color=_pct(win.bench_return), cmin=-lim, cmax=lim, cmid=0, colorscale=[[0, "#b3312f"], [0.5, "#c4c9d4"], [1, "#2f5fc7"]],
                    colorbar=dict(title=dict(text="QQQ return over the window", side="bottom"), orientation="h", thickness=10, lenmode="fraction", len=0.8, x=0.5, xanchor="center", yref="container", y=0.0, yanchor="bottom", ticksuffix="%"))))
    if "actual_return" in win:
        act = (1 + win.actual_return) / (1 + win.bench_return) ** emp.LEVERAGE - 1
        cda = np.column_stack([cd[:, 0], cd[:, 1], _pct(win.bench_return), _pct(win.actual_return), cd[:, 4], _pct(win.actual_return - win.naive_return), cd[:, 6], cd[:, 7]])
        fig.add_trace(go.Scatter(
            x=_pct(win.vol), y=_pct(act), mode="markers", name="Actual TQQQ", visible="legendonly", customdata=cda,
            hovertemplate=ht.replace("Idealized 3× return", "TQQQ actual return"), marker=dict(size=6, symbol="circle-open", color=ORANGE, line=dict(width=1.2))))
    sig = np.linspace(win.vol.min() * 0.95, win.vol.max() * 1.03, 80)
    fig.add_trace(go.Scatter(x=np.round(100 * sig, 3), y=np.round(100 * (np.exp(-emp.LEVERAGE * (emp.LEVERAGE - 1) / 2 * sig**2 * T) - 1), 4),
                             mode="lines", name="Model curve", line=dict(color=INK, width=2, dash="dash"),
                             hovertemplate="Model at %{x:.2f}% volatility: %{y:.2f}%<extra></extra>"))
    fig.update_layout(**BASE, height=520, margin=dict(l=56, r=16, t=16, b=140), hovermode="closest",
                      legend=dict(orientation="v", x=0.99, y=0.99, xanchor="right", yanchor="top", font=dict(size=11), bgcolor="rgba(255,255,255,0.85)"),
                      xaxis=dict(title="Realized volatility (annualized)", ticksuffix="%", gridcolor=GRID, zeroline=False, linecolor=AXIS),
                      yaxis=dict(title="Gap vs (1 + QQQ return)³", ticksuffix="%", gridcolor=GRID, zeroline=True, zerolinecolor=AXIS))
    return fig


# --------------------------------------------------------------------------------------- Figure 6
def fig_green_days(win: pd.DataFrame, horizon: int, edges=(0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0)) -> go.Figure:
    labels = ["≤45", "45–50", "50–55", "55–60", "60–65", ">65"]
    b = pd.cut(win.pos_share, list(edges), include_lowest=True, labels=labels)
    fig = go.Figure()
    for lab in labels:
        g = win[b == lab]
        cd = np.column_stack([[_d(d) for d in g.base_date], [_d(d) for d in g.end_date], _pct(g.bench_return), _pct(g.naive_return), _pct(g.vol), g.pos_days.values])
        fig.add_trace(go.Box(
            y=_pct(g.lev_return), name=f"{lab}<br>n={len(g)}", boxpoints="all", jitter=0.55, pointpos=0, whiskerwidth=0.6, line=dict(color=BLUE, width=1.6),
            fillcolor="rgba(55,88,215,0.12)", marker=dict(size=3.5, opacity=0.45, color=BLUE), boxmean=False, customdata=cd, showlegend=False,
            hovertemplate=("<b>%{customdata[0]} to %{customdata[1]}</b><br>Idealized 3× return: %{y:+.2f}%<br>QQQ return: %{customdata[2]:+.2f}%<br>"
                           "3 × QQQ return: %{customdata[3]:+.2f}%<br>Realized volatility: %{customdata[4]:.2f}%<br>Positive days: %{customdata[5]} of " + str(horizon) + "<extra></extra>")))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1.2, dash="dash"))
    # share of windows in each group whose idealized 3x return is positive (a sample statistic; windows overlap heavily)
    fig.add_annotation(xref="paper", yref="paper", x=0, y=1.13, xanchor="left", yanchor="bottom", showarrow=False, text="Windows ending above zero:", font=dict(size=12, color=MUTED))
    for lab in labels:
        g = win[b == lab]
        fig.add_annotation(xref="x", yref="paper", x=f"{lab}<br>n={len(g)}", y=1.04, xanchor="center", yanchor="bottom", showarrow=False,
                           text=f"<b>{100 * (g.lev_return > 0).mean():.1f}%</b>", font=dict(size=10, color=INK))
    fig.update_layout(**BASE, height=520, margin=dict(l=64, r=16, t=92, b=88), hovermode="closest",
                      xaxis=dict(title="Positive days (% of the window)", linecolor=AXIS, tickangle=0, tickfont=dict(size=11)),
                      yaxis=dict(title="Idealized 3× return", ticksuffix="%", gridcolor=GRID, zeroline=False))
    return fig


# --------------------------------------------------------------------------------------- export
def export(fig: go.Figure, out_dir: Path, name: str, size=(1100, 520), png: bool = True) -> None:
    """Write ``<name>.json`` (Plotly spec, no template) and, if kaleido and Chrome are available, ``<name>.png``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(pio.to_json(fig))
    spec.get("layout", {}).pop("template", None)
    (out_dir / f"{name}.json").write_text(json.dumps(spec, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
    if png:
        try:
            pio.write_image(fig, str(out_dir / f"{name}.png"), width=size[0], height=size[1], scale=2)
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"Static PNG not written for {name} ({type(exc).__name__}); the JSON spec was exported.")
