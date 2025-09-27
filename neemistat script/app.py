# app.py — Neemistat+ v2.3 (Zone-Based Density Map with Real Prices)
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
import time as t

# Optional statsmodels (for AR(1) TR%)
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
.block-container {max-width: 1500px; padding-top: 1rem;}
.main, .block-container { background-color:#0b0c0f; color:#e5e7eb; font-family: Inter, system-ui;}
section[data-testid="stSidebar"]{background:#0f1115; border-right:1px solid #1f2430;}
div[role="radiogroup"] label{background:#151821; border:1px solid #2a3142; border-radius:12px; padding:12px 14px; margin:6px 0; color:#f3f4f6; font-weight:600;}
.hero{padding:48px 16px; background: radial-gradient(1000px 400px at 60% -150px, #1a1f2a 0%, rgba(11,12,15,0) 60%);}
.hero h1{font-size:56px; font-weight:800; color:#e5e7eb;}
.hero h1 .accent{color:#b9c1cc;}
.hero p{color:#9aa3b2;}
.note{font-size:12px; color:#9aa3b2}
</style>
""", unsafe_allow_html=True)

# =========================
# Sidebar controls
# =========================
st.sidebar.markdown("### Neemistat+ V2.3")
mode = st.sidebar.radio("HOD/LOD Mode", ["💲 Price-Based", "🕒 Time-Based", "✔ Level Validator"])
bias_mode = st.sidebar.radio("Bias", ["📊 Prediction", "📈 Forecasting"])
live = st.sidebar.checkbox("🔄 Updates every 2 seconds", value=False)
lookback_years = st.sidebar.number_input("📅 Historical years", min_value=1, max_value=5, value=2)
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
    dt_col = None
    for cand in ["datetime","timestamp","time","date","ts"]:
        if cand in df.columns:
            dt_col = cand; break
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])
    return df, dt_col

@st.cache_data
def make_daily(df, dt_col, price_cols):
    d = df.copy()
    for c in price_cols.values():
        d[c] = pd.to_numeric(d[c], errors="coerce")
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

def density_map_zone_based(df, dt_col, price_cols):
    tmp = df.copy()
    price_col = price_cols.get("close")
    if price_col not in tmp.columns:
        return None, None
    prices = tmp[price_col].dropna()
    if prices.empty: return None, None
    p25, p50, p75 = np.percentile(prices, [25, 50, 75])
    return (p25, p50, p75), prices

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
# File handling
# =========================
if uploaded is not None:
    raw_df, dt_col = read_and_clean(uploaded)
elif use_demo:
    demo_csv = StringIO("""datetime,open,high,low,close
2025-01-02 09:30:00,6060,6065,6055,6062.5
2025-01-02 09:31:00,6062.5,6068,6060,6066.8
2025-01-02 09:32:00,6066.8,6072,6064,6069
2025-01-02 09:33:00,6069,6075,6066,6072
""")
    raw_df, dt_col = read_and_clean(demo_csv)
else:
    st.info("Upload intraday CSV or click **Use Demo Data** in the sidebar.")
    st.stop()

price_cols = {"open":"open","high":"high","low":"low","close":"close"}

# =========================
# Daily OHLC
# =========================
daily = make_daily(raw_df, dt_col, price_cols)
st.subheader("📅 Daily OHLC Sample")
st.dataframe(daily.tail(10))

# =========================
# Zone-Based Density Map (with real MNQ prices)
# =========================
st.subheader("Density Map - Zone Based")
bands, prices = density_map_zone_based(raw_df, dt_col, price_cols)

if bands is not None:
    p25, p50, p75 = bands
    # Use actual quantiles of MNQ prices for levels
    zone_levels = np.quantile(prices, [0.1, 0.25, 0.5, 0.75, 0.9])

    figZ = go.Figure()
    for lvl in zone_levels:
        figZ.add_hline(
            y=lvl,
            line=dict(color="white", width=1, dash="dot"),
            annotation_text=f"{lvl:.2f}",
            annotation_position="right"
        )

    # Median, Inner, Outer
    figZ.add_hline(y=p50, line=dict(color="green", width=2), annotation_text="Median Line")
    figZ.add_hline(y=p25, line=dict(color="white", width=1, dash="dash"), annotation_text="Inner Line")
    figZ.add_hline(y=p75, line=dict(color="white", width=1, dash="dash"), annotation_text="Outer Line")

    figZ.update_layout(
        title="Density Map - Zone Based",
        yaxis_title="MNQ Price",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="white")
    )
    st.plotly_chart(figZ, use_container_width=True)

# =========================
# Forecasting
# =========================
st.subheader("Quantile Forecast")
qdf = quantile_forecast(daily, lookback=lookback_days)
st.dataframe(qdf)

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
