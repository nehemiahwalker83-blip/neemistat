# app.py — Neemistat+ v4.0 (VW-KDE for 1-minute entries)
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
import time as t

# Optional: AR(1) for True Range via statsmodels
try:
    from statsmodels.tsa.ar_model import AutoReg
    STATS_AR_OK = True
except Exception:
    STATS_AR_OK = False

# =========================
# Page config + dark theme
# =========================
st.set_page_config(page_title="Neemistat+", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {max-width: 1500px; padding-top: .5rem;}
.main, .block-container { background-color:#0b0c0f; color:#e5e7eb; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto;}
section[data-testid="stSidebar"]{background:#0f1115; border-right:1px solid #1f2430;}
div[role="radiogroup"] label{background:#151821; border:1px solid #2a3142; border-radius:12px; padding:12px 14px; margin:6px 0; color:#f3f4f6; font-weight:600;}
div[role="radiogroup"] label:hover{border-color:#10b981; background:#121621;}
.hero{padding:34px 16px 14px 16px; background: radial-gradient(1000px 380px at 60% -150px, #1a1f2a 0%, rgba(11,12,15,0) 60%);}
.hero h1{font-size:48px; font-weight:800; color:#e5e7eb; margin:0;}
.hero h1 .accent{color:#b9c1cc;}
.hero p{color:#9aa3b2; margin:.25rem 0;}
.note{font-size:12px; color:#9aa3b2}
</style>
""", unsafe_allow_html=True)

# =========================
# Sidebar controls
# =========================
st.sidebar.markdown("### Neemistat+ v4.0")
st.sidebar.markdown("#### HOD/LOD")
mode = st.sidebar.radio("", ["💲 Price-Based", "🕒 Time-Based", "✔ Level Validator"])

st.sidebar.markdown("#### Bias")
bias_mode = st.sidebar.radio("", ["📊 Prediction", "📈 Forecasting"])

st.sidebar.markdown("#### Options")
live = st.sidebar.checkbox("🔄 Updates every 2 seconds", value=False)
lookback_years = st.sidebar.number_input("📅 Historical years for forecasting", min_value=1, max_value=5, value=2)
lookback_days = int(252 * lookback_years)

uploaded = st.sidebar.file_uploader("📂 Upload intraday CSV", type=["csv"])
use_demo = st.sidebar.button("📊 Use Demo Data")

# =========================
# Hero / heading
# =========================
st.markdown("""
<div class="hero">
  <h1>Experience The Power of <span class="accent">Neemistat+</span></h1>
  <p>Dive into our software and see how we're bringing the power of statistical data to orderflow.</p>
  <p class="note">Note: This software is for informational purposes only and does not constitute financial advice.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
@st.cache_data
def read_and_clean(file, datetime_hint=None):
    df = pd.read_csv(file)
    df.columns = [str(c).strip().lower() for c in df.columns]
    # pick datetime column
    dt_col = None
    if datetime_hint and datetime_hint in df.columns:
        dt_col = datetime_hint
    else:
        for cand in ["datetime","timestamp","time","date","ts"]:
            if cand in df.columns:
                dt_col = cand; break
    if dt_col is None:
        raise ValueError("No datetime-like column found.")
    df[dt_col] = pd.to_datetime(df[dt_col].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=[dt_col]).reset_index(drop=True)
    # trim strings
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()
    return df, dt_col

@st.cache_data
def make_daily(df, dt_col, price_cols):
    d = df.copy()
    for c in price_cols.values():
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=list(price_cols.values()))
    d["date"] = d[dt_col].dt.date
    agg = {
        price_cols["open"]: "first",
        price_cols["high"]: "max",
        price_cols["low"]:  "min",
        price_cols["close"]:"last",
    }
    daily = d.groupby("date").agg(agg)
    daily.index = pd.to_datetime(daily.index)
    daily.columns = ["open","high","low","close"]
    for c in ["open","high","low","close"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")
    return daily.dropna()

def compute_hod_lod_by_bull_bear(daily):
    d = daily.copy()
    d["bullish"] = d["close"] > d["open"]
    return {
        "bullish_highs": d[d["bullish"]]["high"].values,
        "bullish_lows":  d[d["bullish"]]["low"].values,
        "bearish_highs": d[~d["bullish"]]["high"].values,
        "bearish_lows":  d[~d["bullish"]]["low"].values,
    }

def quantile_forecast(daily, lookback=500, quantiles=[0.05,0.25,0.5,0.75,0.95]):
    d = daily.copy()
    d["high_off_open"]  = d["high"] - d["open"]
    d["low_off_open"]   = d["low"]  - d["open"]
    d["close_off_open"] = d["close"]- d["open"]
    last_open = d["open"].iloc[-1]
    q = d[["high_off_open","low_off_open","close_off_open"]].tail(lookback).quantile(quantiles)
    rows = []
    for name in q.index:
        rows.append({
            "quantile":       float(name),
            "forecast_high":  float(last_open + q.loc[name,"high_off_open"]),
            "forecast_low":   float(last_open + q.loc[name,"low_off_open"]),
            "forecast_close": float(last_open + q.loc[name,"close_off_open"]),
        })
    return pd.DataFrame(rows)

def tr_ar1_with_ci(daily, lookback=504):
    if not STATS_AR_OK: return None,None,None
    d = daily.copy()
    trpct = (d["high"] - d["low"]) / d["close"] * 100
    s = trpct.tail(lookback).dropna()
    if len(s) < 3: return None,None,None
    s_log = np.log(s.replace(0,1e-9))
    res = AutoReg(s_log, lags=1, old_names=False).fit()
    pred_log = float(res.predict(start=len(s_log), end=len(s_log)).iloc[0])
    sigma2 = float(getattr(res,"sigma2", np.var(res.resid, ddof=1)))
    se = np.sqrt(sigma2); z = 1.96
    return float(np.exp(pred_log)), float(np.exp(pred_log - z*se)), float(np.exp(pred_log + z*se))

# ===== VW-KDE (no external deps): Silverman bandwidth + volume weights =====
def _silverman_bandwidth(x: np.ndarray):
    # Silverman's rule: 0.9 * min(sd, IQR/1.34) * n^{-1/5}
    n = x.size
    if n < 2: return np.nan
    sd = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    a = min(sd, iqr/1.34) if (sd > 0 and iqr > 0) else (sd if sd>0 else (iqr/1.34 if iqr>0 else 1.0))
    return 0.9 * a * (n ** (-1/5))

def vwkde_price_density(df, price_col="close", weight_col="volume", dt_col="datetime",
                        grid_points=300, scope="session"):
    """
    VW-KDE over real prices for today's session (scope='session').
    Returns:
        grid (y), density (x), levels_dict (median, inner/outer), peaks (list of y's)
    """
    if price_col not in df.columns: return None, None, None, None
    x = pd.to_numeric(df[price_col], errors="coerce")
    tseries = df[dt_col]
    mask = x.notna() & tseries.notna()
    x = x[mask]; tseries = tseries[mask]

    if x.empty: return None, None, None, None

    # Use the latest session (most recent date in data)
    last_day = tseries.dt.date.max()
    if scope == "session":
        sess_mask = tseries.dt.date == last_day
        x = x[sess_mask]

    x = x.values.astype(float)
    n = x.size
    if n < 10: return None, None, None, None

    # volume weights (optional)
    if (weight_col in df.columns):
        w = pd.to_numeric(df.loc[mask, weight_col], errors="coerce")
        if scope == "session":
            w = w[sess_mask]
        w = w.fillna(0).values.astype(float)
        if w.size != n:
            w = None
    else:
        w = None

    # bandwidth
    h = _silverman_bandwidth(x)
    if not np.isfinite(h) or h <= 0:
        # fallback to small fraction of price std
        h = max(np.std(x) * 0.2, 1e-6)

    # evaluation grid
    ymin, ymax = float(np.min(x)), float(np.max(x))
    # expand slightly to avoid cutting tails
    pad = 0.02 * (ymax - ymin if ymax > ymin else max(abs(ymax), 1.0))
    ymin -= pad; ymax += pad
    grid = np.linspace(ymin, ymax, grid_points)

    # VW-KDE: f(x) = 1/(h * sum w) * sum w_i * phi((x - x_i)/h)
    # Vectorized computation
    diffs = (grid[:, None] - x[None, :]) / h
    gauss = np.exp(-0.5 * diffs**2) / np.sqrt(2*np.pi)
    if w is None:
        dens = gauss.mean(axis=1) / h
    else:
        wpos = np.clip(w, 0, None)
        wsum = np.sum(wpos)
        if wsum <= 0:
            dens = gauss.mean(axis=1) / h
        else:
            dens = (gauss @ wpos) / (wsum * h)

    # Normalize density so area under curve ~1 on the grid (approx)
    # (not strictly needed for quantiles since we use CDF over grid)
    dens = np.clip(dens, 0, None)

    # CDF & quantile levels (outer: 5/95, inner: 25/75, median: 50)
    cdf = np.cumsum(dens)
    cdf = cdf / cdf[-1]
    def q_at(p):
        idx = int(np.searchsorted(cdf, p))
        idx = np.clip(idx, 0, grid.size-1)
        return float(grid[idx])

    levels = {
        "outer_low":  q_at(0.05),
        "inner_low":  q_at(0.25),
        "median":     q_at(0.50),
        "inner_high": q_at(0.75),
        "outer_high": q_at(0.95),
    }

    # Peak detection (local maxima above a prominence threshold)
    # Simple discrete derivative test with minimum prominence
    peaks = []
    dens_sm = pd.Series(dens).rolling(3, center=True, min_periods=1).mean().values
    thr = np.percentile(dens_sm, 70)  # only significant peaks
    for i in range(1, len(dens_sm)-1):
        if dens_sm[i] > dens_sm[i-1] and dens_sm[i] > dens_sm[i+1] and dens_sm[i] >= thr:
            peaks.append(float(grid[i]))

    return grid, dens, levels, peaks

# =========================
# File handling
# =========================
if uploaded is not None:
    raw_df, dt_guess = read_and_clean(uploaded)
elif use_demo:
    demo_csv = StringIO("""datetime,open,high,low,close,volume
2025-01-02 09:30:00,6060,6065,6055,6062.50,900
2025-01-02 09:31:00,6062.50,6068,6060,6066.84,1100
2025-01-02 09:32:00,6066.84,6072,6064,6069.10,1040
2025-01-02 09:33:00,6069.10,6075,6066,6072.00,860
2025-01-02 09:34:00,6072.00,6076,6068,6074.50,920
""")
    raw_df, dt_guess = read_and_clean(demo_csv)
else:
    st.info("Upload intraday CSV or click **Use Demo Data** in the sidebar.")
    st.stop()

# Column mapping
cols = raw_df.columns.tolist()
st.sidebar.markdown("#### Column mapping")
def _idx(cols, name, fallback):
    try:
        return cols.index(name)
    except Exception:
        # try first starting with name
        for i,c in enumerate(cols):
            if c.startswith(name): return i
        return cols.index(fallback) if fallback in cols else 0

open_col  = st.sidebar.selectbox("open",  cols, index=_idx(cols,"open","open"))
high_col  = st.sidebar.selectbox("high",  cols, index=_idx(cols,"high","high"))
low_col   = st.sidebar.selectbox("low",   cols, index=_idx(cols,"low","low"))
close_col = st.sidebar.selectbox("close", cols, index=_idx(cols,"close","close"))
dt_col    = st.sidebar.selectbox("datetime", cols, index=_idx(cols, dt_guess, "datetime"))
vol_col   = "volume" if "volume" in cols else None

price_cols = {"open":open_col, "high":high_col, "low":low_col, "close":close_col}

# Level Validator input
levels = []
if mode == "✔ Level Validator":
    lv_str = st.sidebar.text_input("Levels (comma separated)", value="")
    if lv_str.strip():
        try:
            levels = [float(x.strip()) for x in lv_str.split(",") if x.strip()!=""]
        except Exception:
            st.sidebar.error("Levels must be numeric, comma-separated.")

# =========================
# Daily OHLC
# =========================
with st.spinner("Aggregating daily OHLC…"):
    daily = make_daily(raw_df, dt_col, price_cols)
if daily.empty:
    st.error("No valid daily OHLC rows after cleaning. Check mapping/CSV.")
    st.stop()

st.subheader("📅 Daily OHLC — sample")
st.dataframe(daily.tail(10))

# =========================
# HOD/LOD (Bullish vs Bearish)
# =========================
st.subheader("HOD / LOD (Bullish vs Bearish)")
cA, cB = st.columns(2)
hodlod = compute_hod_lod_by_bull_bear(daily)
with cA:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=hodlod["bullish_highs"], name="Bullish HOD", nbinsx=50, opacity=0.75))
    fig.add_trace(go.Histogram(x=hodlod["bearish_highs"], name="Bearish HOD", nbinsx=50, opacity=0.55))
    fig.update_layout(barmode="overlay", title="High of Day distribution", paper_bgcolor="#0b0c0f", plot_bgcolor="#0b0c0f", font=dict(color="#e5e7eb"))
    st.plotly_chart(fig, use_container_width=True)
with cB:
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=hodlod["bullish_lows"], name="Bullish LOD", nbinsx=50, opacity=0.75))
    fig2.add_trace(go.Histogram(x=hodlod["bearish_lows"], name="Bearish LOD", nbinsx=50, opacity=0.55))
    fig2.update_layout(barmode="overlay", title="Low of Day distribution", paper_bgcolor="#0b0c0f", plot_bgcolor="#0b0c0f", font=dict(color="#e5e7eb"))
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# Density Map — VW-KDE (session-only, volume-weighted, shaded bands + peaks)
# =========================
st.subheader("Density Map — VW-KDE (Session, Volume-Weighted)")

grid, dens, levels_dict, peaks = vwkde_price_density(
    raw_df.rename(columns={close_col:"close", dt_col:"datetime", (vol_col or "volume"):(vol_col or "volume")}),
    price_col="close",
    weight_col=(vol_col or "volume"),
    dt_col="datetime",
    grid_points=400,
    scope="session"
)

if grid is not None:
    median = levels_dict["median"]
    il, ih = levels_dict["inner_low"], levels_dict["inner_high"]
    ol, oh = levels_dict["outer_low"], levels_dict["outer_high"]

    # Plot: density curve (horizontal), with shaded inner & outer bands, median, peaks
    figZ = go.Figure()

    # KDE density curve
    figZ.add_trace(go.Scatter(
        x=dens, y=grid, mode="lines", name="VW-KDE",
        line=dict(color="#9ca3af", width=2)
    ))
    # Fill under curve
    figZ.add_trace(go.Scatter(
        x=dens, y=grid, fill='tozerox', mode='none',
        fillcolor="rgba(156,163,175,0.20)", name="Density"
    ))

    # Shaded bands: outer (5–95) & inner (25–75)
    figZ.add_hrect(y0=ol, y1=oh, line_width=0, fillcolor="rgba(255,255,255,0.05)", layer="below")
    figZ.add_hrect(y0=il, y1=ih, line_width=0, fillcolor="rgba(34,197,94,0.10)", layer="below")

    # Lines: outer/inner/median
    figZ.add_hline(y=median, line=dict(color="#22c55e", width=2), annotation_text=f"Median {median:.2f}", annotation_position="left")
    figZ.add_hline(y=il, line=dict(color="white", width=1, dash="dash"), annotation_text=f"Inner {il:.2f}", annotation_position="left")
    figZ.add_hline(y=ih, line=dict(color="white", width=1, dash="dash"), annotation_text=f"Inner {ih:.2f}", annotation_position="left")
    figZ.add_hline(y=ol, line=dict(color="white", width=1, dash="dot"),  annotation_text=f"Outer {ol:.2f}", annotation_position="left")
    figZ.add_hline(y=oh, line=dict(color="white", width=1, dash="dot"),  annotation_text=f"Outer {oh:.2f}", annotation_position="left")

    # Peaks (modes)
    if peaks:
        figZ.add_trace(go.Scatter(
            x=[np.interp(p, grid, dens) for p in peaks],
            y=peaks,
            mode="markers",
            name="Modes",
            marker=dict(size=8, color="#60a5fa", line=dict(color="white", width=1))
        ))

    figZ.update_layout(
        title="VW-KDE Price Density (Session) — Inner/Outer Bands + Median + Modes",
        xaxis_title="Density (volume-weighted KDE)", yaxis_title="Price",
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        font=dict(color="white"),
        showlegend=True
    )
    st.plotly_chart(figZ, use_container_width=True)

    # Numeric levels table (for precision entries)
    levels_df = pd.DataFrame([{
        "Outer Low (5%)": ol, "Inner Low (25%)": il,
        "Median (50%)": median,
        "Inner High (75%)": ih, "Outer High (95%)": oh
    }]).T.rename(columns={0:"Price"})
    st.dataframe(levels_df.style.format({"Price":"{:.2f}"}))
else:
    st.info("Not enough data to compute VW-KDE for the latest session.")

# =========================
# Modes: Level Validator
# =========================
if mode == "✔ Level Validator":
    st.subheader("Level Validator")
    if not levels:
        st.info("Enter levels in the sidebar to compute historical hit rates.")
    else:
        recent = daily.tail(lookback_days)
        rows = []
        for lvl in levels:
            touched = ((recent["low"] <= lvl) & (recent["high"] >= lvl)).sum()
            pct = (touched / len(recent) * 100.0) if len(recent) else np.nan
            rows.append({"level": lvl, "hit_rate_%": pct})
        st.dataframe(pd.DataFrame(rows))

# =========================
# Forecasting (default 2Y lookback)
# =========================
st.subheader("Quantile Forecast (applied to last open)")
qdf = quantile_forecast(daily, lookback=lookback_days, quantiles=[0.05,0.25,0.5,0.75,0.95])
if qdf is None or qdf.empty:
    st.info("Insufficient history for quantile forecast.")
else:
    st.dataframe(qdf)
    figQ = px.line(qdf, x="quantile", y=["forecast_high","forecast_close","forecast_low"], markers=True)
    figQ.update_layout(title="Quantile Bands", paper_bgcolor="#0b0c0f", plot_bgcolor="#0b0c0f", font=dict(color="#e5e7eb"))
    st.plotly_chart(figQ, use_container_width=True)

# =========================
# Next-day True Range (percent & points)
# =========================
st.subheader("Next-day True Range")
if STATS_AR_OK:
    pred_pct, lo_pct, hi_pct = tr_ar1_with_ci(daily, lookback=lookback_days)
    if pred_pct is None:
        tr_hist = (daily["high"] - daily["low"]).div(daily["close"])*100
        pred_pct = float(tr_hist.iloc[-1]); lo_pct = hi_pct = None
else:
    tr_hist = (daily["high"] - daily["low"]).div(daily["close"])*100
    pred_pct = float(tr_hist.iloc[-1]); lo_pct = hi_pct = None

last_close = float(daily["close"].iloc[-1])
pred_points = last_close * (pred_pct/100.0)
c1, c2 = st.columns(2)
c1.metric("TR (percent)", f"{pred_pct:.3f}%")
c2.metric("TR (points)", f"{pred_points:.4f}")

# TR history + forecast marker
tr_hist = (daily["high"] - daily["low"]).div(daily["close"])*100
figTR = go.Figure()
figTR.add_trace(go.Scatter(x=tr_hist.index, y=tr_hist.values, mode="lines", name="Historical TR%"))
next_date = tr_hist.index[-1] + pd.Timedelta(days=1)
figTR.add_trace(go.Scatter(x=[next_date], y=[pred_pct], mode="markers", name="Forecast", marker=dict(size=10)))
if lo_pct is not None and hi_pct is not None:
    figTR.add_trace(go.Scatter(x=[next_date, next_date], y=[lo_pct, hi_pct], mode="lines+markers", name="95% CI"))
figTR.update_layout(title="True Range % with Next-day Forecast", yaxis_title="TR %", paper_bgcolor="#0b0c0f", plot_bgcolor="#0b0c0f", font=dict(color="#e5e7eb"))
st.plotly_chart(figTR, use_container_width=True)

# =========================
# Live refresh
# =========================
if live:
    t.sleep(2)
    st.experimental_rerun()
