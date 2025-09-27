# app.py — Neemistat+ v2.0 (styled)
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
.main, .block-container { background-color:#0b0c0f; color:#e5e7eb; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";}
a {color:#10b981 !important}

/* sidebar */
section[data-testid="stSidebar"]{background:#0f1115; border-right:1px solid #1f2430;}
section[data-testid="stSidebar"] .stRadio, section[data-testid="stSidebar"] .stSelectbox, section[data-testid="stSidebar"] .stNumberInput {margin-bottom: 0.6rem;}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3{color:#fff; margin: 0.2rem 0 0.6rem 0;}
/* button-like radios */
div[role="radiogroup"] label{background:#151821; border:1px solid #2a3142; border-radius:12px; padding:12px 14px; margin:6px 0; color:#f3f4f6; font-weight:600; cursor:pointer;}
div[role="radiogroup"] label:hover{border-color:#10b981; background:#121621;}
div[role="radiogroup"] input:checked + div{color:#0b0c0f}
div[role="radiogroup"] label[data-baseweb="radio"] {box-shadow: 0 2px 10px rgba(0,0,0,0.25);}

/* title hero */
.hero{padding:48px 16px 28px 16px; text-align:left; background: radial-gradient(1000px 400px at 60% -150px, #1a1f2a 0%, rgba(11,12,15,0) 60%);}
.hero h1{font-size:56px; line-height:1.05; margin:0; letter-spacing:-0.02em; color:#e5e7eb;}
.hero h1 .accent{color:#b9c1cc;}
.hero p{color:#9aa3b2; margin-top:10px}
.note{font-size:12px; color:#9aa3b2}

/* plotly theme */
.plotly .main-svg{background:#0b0c0f !important}
</style>
""", unsafe_allow_html=True)

# =========================
# Sidebar: controls
# =========================
st.sidebar.markdown("### QuantiStat V2.0")  # label like your screenshot
st.sidebar.markdown("#### HOD/LOD")
mode = st.sidebar.radio("", ["💲 Price-Based", "🕒 Time-Based", "✔ Level Validator"])

st.sidebar.markdown("#### Bias")
bias_mode = st.sidebar.radio("", ["📊 Prediction", "📈 Forecasting"])

st.sidebar.markdown("#### Options")
live = st.sidebar.checkbox("🔄 Updates every 2 seconds", value=False)
lookback_years = st.sidebar.number_input("📅 Historical years for forecasting", min_value=1, max_value=5, value=2)
lookback_days = int(252 * lookback_years)

uploaded = st.sidebar.file_uploader("📂 Upload intraday CSV", type=["csv"])
st.sidebar.markdown("---")

# Column mapping UI appears after upload
def _safe_index(lst, pred, default=0):
    try:
        return lst.index(next((c for c in lst if pred(c)), lst[0]))
    except Exception:
        return default

# =========================
# Hero / heading
# =========================
st.markdown(
    """
<div class="hero">
  <h1>Experience The Power of <span class="accent">Neemistat+</span></h1>
  <p>Dive into our software and see how we're bringing the power of statistical data to orderflow.</p>
  <p class="note">Note: This software is for informational purposes only and does not constitute financial advice. Please consult a professional before making investment decisions.</p>
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
    # strip surrounding whitespace
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
        price_cols["low"]: "min",
        price_cols["close"]: "last",
    }
    daily = d.groupby("date").agg(agg)
    daily.index = pd.to_datetime(daily.index)
    daily.columns = ["open","high","low","close"]
    for c in ["open","high","low","close"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")
    return daily.dropna()

def density_map_price_based(df, dt_col, price_cols, bins_time=200, bins_price=150):
    tmp = df.copy()
    price_col = price_cols.get("close") if price_cols.get("close") in tmp.columns else price_cols.get("open")
    tmp["tod"] = tmp[dt_col].dt.hour*3600 + tmp[dt_col].dt.minute*60 + tmp[dt_col].dt.second
    tmp["date_only"] = tmp[dt_col].dt.date
    opens = tmp.groupby("date_only")[price_cols["open"]].first().rename("day_open")
    tmp = tmp.merge(opens, left_on="date_only", right_index=True, how="left")
    tmp["price_offset"] = tmp[price_col] - tmp["day_open"]
    x = tmp["tod"].values
    y = tmp["price_offset"].values
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]; y = y[mask]
    if len(x) == 0:
        return None, None, None, None
    H, xedges, yedges = np.histogram2d(x, y, bins=[bins_time, bins_price])
    # inner/median/outer = p25 / p50 / p75 of price_offset
    p25, p50, p75 = np.nanpercentile(tmp["price_offset"].dropna(), [25,50,75]) if tmp["price_offset"].notna().any() else (None,None,None)
    return H.T, xedges, yedges, (p25,p50,p75)

def compute_hod_lod_by_bull_bear(daily):
    d = daily.copy()
    d["bullish"] = d["close"] > d["open"]
    bullish = d[d["bullish"]]
    bearish = d[~d["bullish"]]
    return {
        "bullish_highs": bullish["high"].values,
        "bullish_lows": bullish["low"].values,
        "bearish_highs": bearish["high"].values,
        "bearish_lows": bearish["low"].values,
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
    if not STATS_OK:
        return None, None, None
    d = daily.copy()
    trpct = (d["high"] - d["low"]) / d["close"] * 100
    s = trpct.tail(lookback).dropna()
    if len(s) < 3:
        return None, None, None
    s_log = np.log(s.replace(0, 1e-9))
    res = AutoReg(s_log, lags=1, old_names=False).fit()
    pred_log = float(res.predict(start=len(s_log), end=len(s_log)).iloc[0])
    sigma2 = float(getattr(res, "sigma2", np.var(res.resid, ddof=1)))
    se = np.sqrt(sigma2); z = 1.96
    lower, upper = np.exp(pred_log - z*se), np.exp(pred_log + z*se)
    return float(np.exp(pred_log)), float(lower), float(upper)

# =========================
# Guard: need file
# =========================
if uploaded is None:
    st.info("Upload intraday CSV with columns: datetime, open, high, low, close.")
    st.stop()

# Column mapping (so any header names work)
raw_df, dt_guess = read_and_clean(uploaded)
cols = raw_df.columns.tolist()
st.sidebar.markdown("#### Column mapping")
open_col  = st.sidebar.selectbox("open",  cols, index=_safe_index(cols, lambda c: c.lower().startswith("open")))
high_col  = st.sidebar.selectbox("high",  cols, index=_safe_index(cols, lambda c: c.lower().startswith("high")))
low_col   = st.sidebar.selectbox("low",   cols, index=_safe_index(cols, lambda c: c.lower().startswith("low")))
close_col = st.sidebar.selectbox("close", cols, index=_safe_index(cols, lambda c: c.lower().startswith("close")))
dt_col    = st.sidebar.selectbox("datetime", cols, index=_safe_index(cols, lambda c: c == dt_guess))
price_cols = {"open": open_col, "high": high_col, "low": low_col, "close": close_col}

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
# Build daily OHLC
# =========================
with st.spinner("Aggregating daily OHLC…"):
    daily = make_daily(raw_df, dt_col, price_cols)

if daily.empty:
    st.error("No valid daily OHLC rows after cleaning. Check column mapping or CSV format.")
    st.stop()

# =========================
# HOD/LOD (bullish vs bearish)
# =========================
st.subheader("HOD / LOD (Bullish vs Bearish)")
ca, cb = st.columns(2)

hodlod = compute_hod_lod_by_bull_bear(daily)

with ca:
    figH = go.Figure()
    figH.add_trace(go.Histogram(x=hodlod["bullish_highs"], name="Bullish HOD", nbinsx=50, opacity=0.75))
    figH.add_trace(go.Histogram(x=hodlod["bearish_highs"], name="Bearish HOD", nbinsx=50, opacity=0.55))
    figH.update_layout(barmode="overlay", title="High-of-Day Distribution", paper_bgcolor="#0b0c0f", plot_bgcolor="#0b0c0f", font=dict(color="#e5e7eb"))
    st.plotly_chart(figH, use_container_width=True)
with cb:
    figL = go.Figure()
    figL.add_trace(go.Histogram(x=hodlod["bullish_lows"], name="Bullish LOD", nbinsx=50, opacity=0.75))
    figL.add_trace(go.Histogram(x=hodlod["bearish_lows"], name="Bearish LOD", nbinsx=50, opacity=0.55))
    figL.update_layout(barmode="overlay", title="Low-of-Day Distribution", paper_bgcolor="#0b0c0f", plot_bgcolor="#0b0c0f", font=dict(color="#e5e7eb"))
    st.plotly_chart(figL, use_container_width=True)

# =========================
# Density map (Price-Based / Time-Based)  with inner/median/outer lines
# =========================
st.subheader("Density Map")
if mode in ["💲 Price-Based", "🕒 Time-Based"]:
    H, xedges, yedges, bands = density_map_price_based(raw_df, dt_col, price_cols, bins_time=200, bins_price=150)
    if H is None:
        st.info("Not enough intraday rows to compute density map.")
    else:
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        p25, p50, p75 = bands
        heat = go.Figure(go.Heatmap(z=H, x=xcenters, y=ycenters, colorscale="Viridis", colorbar=dict(title="Count")))
        if p50 is not None:
            heat.add_hline(y=p50, line=dict(color="white", width=2), name="Median")
        if p25 is not None and p75 is not None:
            heat.add_hline(y=p25, line=dict(color="white", width=1, dash="dash"), name="Inner")
            heat.add_hline(y=p75, line=dict(color="white", width=1, dash="dash"), name="Outer")

        if mode == "🕒 Time-Based":
            xticks = [int(x) for x in np.linspace(xcenters.min(), xcenters.max(), 6)]
            xtlabels = []
            for s in xticks:
                hh = int(s//3600); mm = int((s%3600)//60)
                xtlabels.append(f"{hh:02d}:{mm:02d}")
            heat.update_layout(xaxis=dict(tickmode="array", tickvals=xticks, ticktext=xtlabels))

        heat.update_layout(title="Time vs Price Offset", paper_bgcolor="#0b0c0f", plot_bgcolor="#0b0c0f", font=dict(color="#e5e7eb"))
        st.plotly_chart(heat, use_container_width=True)
else:
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
# Forecasting using 2 years (default) — Quantile forecast off last open
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
if STATS_OK:
    pred_pct, lo_pct, hi_pct = tr_ar1_with_ci(daily, lookback=lookback_days)
    if pred_pct is None:
        st.info("Not enough history for AR(1). Using last TR% as persistence.")
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
# Candlestick (for quick context)
# =========================
st.subheader("Daily Candlestick")
dp = daily.reset_index().rename(columns={"index":"date"})
figC = go.Figure(data=[go.Candlestick(x=dp["date"], open=dp["open"], high=dp["high"], low=dp["low"], close=dp["close"])])
figC.update_layout(title="Daily OHLC", xaxis_title="Date", paper_bgcolor="#0b0c0f", plot_bgcolor="#0b0c0f", font=dict(color="#e5e7eb"))
st.plotly_chart(figC, use_container_width=True)

# =========================
# Live updates (every 2s)
# =========================
if live:
    t.sleep(2)
    st.experimental_rerun()

st.markdown("---")
st.caption("Neemistat+ — Density map (inner/median/outer), HOD/LOD split, 2yr forecasting, Bias + Modes, Next-day TR%, Live refresh.")
