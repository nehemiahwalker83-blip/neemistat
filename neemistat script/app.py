# app.py — Neemistat+ v2.1 (with Demo Data fallback)
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import io, zipfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time as t

# Optional statsmodels (for AR(1) TR%); app still works without it
try:
    from statsmodels.tsa.ar_model import AutoReg
    STATS_OK = True
except Exception:
    STATS_OK = False

# =========================
# Page config + dark theme
# =========================
st.set_page_config(page_title="Neemistat+", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* base */
.block-container {max-width: 1500px; padding-top: 1rem;}
.main, .block-container { background-color:#0b0c0f; color:#e5e7eb; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial;}
/* sidebar */
section[data-testid="stSidebar"]{background:#0f1115; border-right:1px solid #1f2430;}
section[data-testid="stSidebar"] h2, h3{color:#fff;}
/* button-like radios */
div[role="radiogroup"] label{background:#151821; border:1px solid #2a3142; border-radius:12px; padding:12px 14px; margin:6px 0; color:#f3f4f6; font-weight:600; cursor:pointer;}
div[role="radiogroup"] label:hover{border-color:#10b981; background:#121621;}
/* hero */
.hero{padding:48px 16px 28px 16px; background: radial-gradient(1000px 400px at 60% -150px, #1a1f2a 0%, rgba(11,12,15,0) 60%);}
.hero h1{font-size:56px; font-weight:800; color:#e5e7eb;}
.hero h1 .accent{color:#b9c1cc;}
.hero p{color:#9aa3b2;}
.note{font-size:12px; color:#9aa3b2}
</style>
""", unsafe_allow_html=True)

# =========================
# Sidebar controls
# =========================
st.sidebar.markdown("### Neemistat+ V2.1")
st.sidebar.markdown("#### HOD/LOD")
mode = st.sidebar.radio("", ["💲 Price-Based", "🕒 Time-Based", "✔ Level Validator"])

st.sidebar.markdown("#### Bias")
bias_mode = st.sidebar.radio("", ["📊 Prediction", "📈 Forecasting"])

st.sidebar.markdown("#### Options")
live = st.sidebar.checkbox("🔄 Updates every 2 seconds", value=False)
lookback_years = st.sidebar.number_input("📅 Historical years", min_value=1, max_value=5, value=2)
lookback_days = int(252 * lookback_years)

uploaded = st.sidebar.file_uploader("📂 Upload intraday CSV", type=["csv"])
use_demo = st.sidebar.button("📊 Use Demo Data")

# =========================
# Hero / heading
# =========================
st.markdown(
    """
<div class="hero">
  <h1>Experience The Power of <span class="accent">Neemistat+</span></h1>
  <p>Dive into our software and see how we're bringing the power of statistical data to orderflow.</p>
  <p class="note">Note: This software is for informational purposes only and does not constitute financial advice.</p>
</div>
""",
    unsafe_allow_html=True,
)

# =========================
# Helpers
# =========================
@st.cache_data
def read_and_clean(file, datetime_hint=None):
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]
    dt_col = None
    for cand in ["datetime","timestamp","time","date","ts"]:
        if cand in df.columns:
            dt_col = cand; break
    if dt_col is None:
        raise ValueError("No datetime-like column found.")
    df[dt_col] = pd.to_datetime(df[dt_col].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=[dt_col]).reset_index(drop=True)
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
        price_cols["low"]: "min",
        price_cols["close"]: "last",
    }
    daily = d.groupby("date").agg(agg)
    daily.index = pd.to_datetime(daily.index)
    daily.columns = ["open","high","low","close"]
    return daily.dropna()

def density_map_price_based(df, dt_col, price_cols, bins_time=200, bins_price=150):
    tmp = df.copy()
    price_col = price_cols.get("close") if price_cols.get("close") in tmp.columns else price_cols.get("open")
    tmp["tod"] = tmp[dt_col].dt.hour*3600 + tmp[dt_col].dt.minute*60 + tmp[dt_col].dt.second
    tmp["date_only"] = tmp[dt_col].dt.date
    opens = tmp.groupby("date_only")[price_cols["open"]].first().rename("day_open")
    tmp = tmp.merge(opens, left_on="date_only", right_index=True, how="left")
    tmp["price_offset"] = tmp[price_col] - tmp["day_open"]
    x, y = tmp["tod"].values, tmp["price_offset"].values
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) == 0: return None,None,None,None
    H, xedges, yedges = np.histogram2d(x, y, bins=[bins_time, bins_price])
    p25,p50,p75 = np.nanpercentile(tmp["price_offset"].dropna(), [25,50,75])
    return H.T, xedges, yedges, (p25,p50,p75)

def compute_hod_lod_by_bull_bear(daily):
    d = daily.copy()
    d["bullish"] = d["close"] > d["open"]
    return {
        "bullish_highs": d[d["bullish"]]["high"].values,
        "bullish_lows": d[d["bullish"]]["low"].values,
        "bearish_highs": d[~d["bullish"]]["high"].values,
        "bearish_lows": d[~d["bullish"]]["low"].values,
    }

def quantile_forecast(daily, lookback=500, quantiles=[0.05,0.25,0.5,0.75,0.95]):
    d = daily.copy()
    d["high_off_open"] = d["high"] - d["open"]
    d["low_off_open"] = d["low"] - d["open"]
    d["close_off_open"] = d["close"] - d["open"]
    last_open = d["open"].iloc[-1]
    q = d[["high_off_open","low_off_open","close_off_open"]].tail(lookback).quantile(quantiles)
    rows = []
    for name in q.index:
        rows.append({
            "quantile": float(name),
            "forecast_high": float(last_open + q.loc[name,"high_off_open"]),
            "forecast_low": float(last_open + q.loc[name,"low_off_open"]),
            "forecast_close": float(last_open + q.loc[name,"close_off_open"]),
        })
    return pd.DataFrame(rows)

def tr_ar1_with_ci(daily, lookback=504):
    if not STATS_OK: return None,None,None
    d = daily.copy()
    trpct = (d["high"] - d["low"]) / d["close"] * 100
    s = trpct.tail(lookback).dropna()
    if len(s) < 3: return None,None,None
    s_log = np.log(s.replace(0,1e-9))
    res = AutoReg(s_log, lags=1, old_names=False).fit()
    pred_log = float(res.predict(start=len(s_log), end=len(s_log)).iloc[0])
    sigma2 = float(getattr(res,"sigma2", np.var(res.resid, ddof=1)))
    se = np.sqrt(sigma2); z=1.96
    return float(np.exp(pred_log)), float(np.exp(pred_log-z*se)), float(np.exp(pred_log+z*se))

# =========================
# File handling: upload or demo
# =========================
if uploaded is not None:
    raw_df, dt_guess = read_and_clean(uploaded)
elif use_demo:
    from io import StringIO
    demo_csv = StringIO("""datetime,open,high,low,close,volume,buy_volume,sell_volume
2025-01-02 09:30:00,17450,17460,17440,17455,100,60,40
2025-01-02 09:31:00,17455,17470,17450,17465,120,70,50
2025-01-02 09:32:00,17465,17480,17460,17475,150,80,70
2025-01-02 09:33:00,17475,17485,17470,17480,130,60,70
2025-01-02 09:34:00,17480,17490,17475,17485,140,90,50
2025-01-02 09:35:00,17485,17495,17480,17490,160,100,60
2025-01-02 09:36:00,17490,17500,17485,17495,180,110,70
""")
    raw_df, dt_guess = read_and_clean(demo_csv)
else:
    st.info("Upload intraday CSV or click **Use Demo Data** in the sidebar.")
    st.stop()

# =========================
# Column mapping
# =========================
cols = raw_df.columns.tolist()
st.sidebar.markdown("#### Column mapping")
open_col = st.sidebar.selectbox("open", cols, index=cols.index("open"))
high_col = st.sidebar.selectbox("high", cols, index=cols.index("high"))
low_col = st.sidebar.selectbox("low", cols, index=cols.index("low"))
close_col = st.sidebar.selectbox("close", cols, index=cols.index("close"))
dt_col = st.sidebar.selectbox("datetime", cols, index=cols.index(dt_guess))

price_cols = {"open": open_col, "high": high_col, "low": low_col, "close": close_col}

# =========================
# Daily OHLC
# =========================
daily = make_daily(raw_df, dt_col, price_cols)
if daily.empty:
    st.error("No valid OHLC rows after cleaning.")
    st.stop()

st.subheader("📅 Daily OHLC Sample")
st.dataframe(daily.tail(10))

# =========================
# HOD/LOD
# =========================
st.subheader("HOD / LOD (Bullish vs Bearish)")
ca, cb = st.columns(2)
hodlod = compute_hod_lod_by_bull_bear(daily)
with ca:
    figH = go.Figure()
    figH.add_trace(go.Histogram(x=hodlod["bullish_highs"], name="Bullish HOD", nbinsx=50))
    figH.add_trace(go.Histogram(x=hodlod["bearish_highs"], name="Bearish HOD", nbinsx=50))
    figH.update_layout(barmode="overlay")
    st.plotly_chart(figH, use_container_width=True)
with cb:
    figL = go.Figure()
    figL.add_trace(go.Histogram(x=hodlod["bullish_lows"], name="Bullish LOD", nbinsx=50))
    figL.add_trace(go.Histogram(x=hodlod["bearish_lows"], name="Bearish LOD", nbinsx=50))
    figL.update_layout(barmode="overlay")
    st.plotly_chart(figL, use_container_width=True)

# =========================
# Density map
# =========================
st.subheader("Density Map")
H, xedges, yedges, bands = density_map_price_based(raw_df, dt_col, price_cols)
if H is not None:
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    p25,p50,p75 = bands
    heat = go.Figure(go.Heatmap(z=H, x=xcenters, y=ycenters, colorscale="Viridis"))
    heat.add_hline(y=p50, line=dict(color="white", width=2))
    heat.add_hline(y=p25, line=dict(color="white", width=1, dash="dash"))
    heat.add_hline(y=p75, line=dict(color="white", width=1, dash="dash"))
    st.plotly_chart(heat, use_container_width=True)

# =========================
# Forecasting
# =========================
st.subheader("Quantile Forecast")
qdf = quantile_forecast(daily, lookback=lookback_days)
st.dataframe(qdf)
figQ = px.line(qdf, x="quantile", y=["forecast_high","forecast_close","forecast_low"], markers=True)
st.plotly_chart(figQ, use_container_width=True)

# =========================
# Next-day TR
# =========================
st.subheader("Next-day True Range")
pred_pct, lo, hi = tr_ar1_with_ci(daily, lookback=lookback_days) if STATS_OK else (None,None,None)
if pred_pct is None:
    tr_hist = (daily["high"] - daily["low"]).div(daily["close"])*100
    pred_pct = float(tr_hist.iloc[-1]); lo=hi=None
last_close = float(daily["close"].iloc[-1])
pred_points = last_close*(pred_pct/100)
c1,c2 = st.columns(2)
c1.metric("TR %", f"{pred_pct:.2f}%")
c2.metric("TR pts", f"{pred_points:.2f}")

# =========================
# Live refresh
# =========================
if live:
    t.sleep(2)
    st.experimental_rerun()
