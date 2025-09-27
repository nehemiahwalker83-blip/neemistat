# app.py — Neemistat+ v3.0
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
import time as t

# Optional statsmodels (AR(1) for TR%)
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
.block-container {max-width: 1500px; padding-top: 0.5rem;}
.main, .block-container { background-color:#0b0c0f; color:#e5e7eb; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto;}
section[data-testid="stSidebar"]{background:#0f1115; border-right:1px solid #1f2430;}
/* button-like radios */
div[role="radiogroup"] label{background:#151821; border:1px solid #2a3142; border-radius:12px; padding:12px 14px; margin:6px 0; color:#f3f4f6; font-weight:600;}
div[role="radiogroup"] label:hover{border-color:#10b981; background:#121621;}
/* hero */
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
st.sidebar.markdown("### Neemistat+")
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
    # trim objects
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
    d["low_off_open"]  = d["low"]  - d["open"]
    d["close_off_open"]= d["close"]- d["open"]
    last_open = d["open"].iloc[-1]
    q = d[["high_off_open","low_off_open","close_off_open"]].tail(lookback).quantile(quantiles)
    rows = []
    for name in q.index:
        rows.append({
            "quantile": float(name),
            "forecast_high":  float(last_open + q.loc[name,"high_off_open"]),
            "forecast_low":   float(last_open + q.loc[name,"low_off_open"]),
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
    se = np.sqrt(sigma2); z = 1.96
    return float(np.exp(pred_log)), float(np.exp(pred_log - z*se)), float(np.exp(pred_log + z*se))

# ---- DENSITY (QuantiStat-style: price occurrence histogram → bands)
def density_map_from_intraday(df, price_col="close", bins=160):
    """
    Build a horizontal histogram of intraday prices and extract band lines:
    - median (50%), inner (25/75%), outer (5/95%) using CDF over price levels.
    Returns centers, density, (median, inner_low, inner_high, outer_low, outer_high).
    """
    if price_col not in df.columns: return None, None, None
    prices = pd.to_numeric(df[price_col], errors="coerce").dropna()
    if prices.empty: return None, None, None

    # Histogram of price occurrences (NOT normalized by time)
    hist, edges = np.histogram(prices, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    # CDF across ascending price
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]

    def level_at(p):
        idx = np.searchsorted(cdf, p)
        idx = np.clip(idx, 0, len(centers)-1)
        return float(centers[idx])

    median_line   = level_at(0.50)
    inner_low     = level_at(0.25)
    inner_high    = level_at(0.75)
    outer_low     = level_at(0.05)
    outer_high    = level_at(0.95)

    return centers, hist, (median_line, inner_low, inner_high, outer_low, outer_high)

# =========================
# File handling
# =========================
if uploaded is not None:
    raw_df, dt_guess = read_and_clean(uploaded)
elif use_demo:
    demo_csv = StringIO("""datetime,open,high,low,close
2025-01-02 09:30:00,6060,6065,6055,6062.50
2025-01-02 09:31:00,6062.50,6068,6060,6066.84
2025-01-02 09:32:00,6066.84,6072,6064,6069.10
2025-01-02 09:33:00,6069.10,6075,6066,6072.00
2025-01-02 09:34:00,6072.00,6076,6068,6074.50
""")
    raw_df, dt_guess = read_and_clean(demo_csv)
else:
    st.info("Upload intraday CSV or click **Use Demo Data** in the sidebar.")
    st.stop()

# Column mapping
cols = raw_df.columns.tolist()
st.sidebar.markdown("#### Column mapping")
open_col  = st.sidebar.selectbox("open",  cols, index=cols.index(next((c for c in cols if c.startswith("open")),  "open")))
high_col  = st.sidebar.selectbox("high",  cols, index=cols.index(next((c for c in cols if c.startswith("high")),  "high")))
low_col   = st.sidebar.selectbox("low",   cols, index=cols.index(next((c for c in cols if c.startswith("low")),   "low")))
close_col = st.sidebar.selectbox("close", cols, index=cols.index(next((c for c in cols if c.startswith("close")), "close")))
dt_col    = st.sidebar.selectbox("datetime", cols, index=cols.index(dt_guess))

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
# Daily OHLC
# =========================
with st.spinner("Aggregating daily OHLC…"):
    daily = make_daily(raw_df, dt_col, price_cols)
if daily.empty:
    st.error("No valid daily OHLC rows after cleaning. Check mapping or CSV.")
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
# Density Map — QuantiStat style (horizontal histogram + bands)
# =========================
st.subheader("Density Map — Zone Based")
centers, density, bands = density_map_from_intraday(raw_df.rename(columns={close_col:"close"}), price_col="close", bins=160)
if centers is not None:
    median_line, inner_low, inner_high, outer_low, outer_high = bands

    figZ = go.Figure()

    # Horizontal histogram of price occurrence
    figZ.add_trace(go.Bar(
        y=centers, x=density, orientation="h",
        marker=dict(color="#6b7280"), opacity=0.55, name="Density"
    ))

    # Band lines at actual MNQ prices
    figZ.add_hline(y=median_line, line=dict(color="#22c55e", width=2), annotation_text=f"Median {median_line:.2f}", annotation_position="left")
    figZ.add_hline(y=inner_low,  line=dict(color="white", width=1, dash="dash"), annotation_text=f"Inner {inner_low:.2f}", annotation_position="left")
    figZ.add_hline(y=inner_high, line=dict(color="white", width=1, dash="dash"), annotation_text=f"Inner {inner_high:.2f}", annotation_position="left")
    figZ.add_hline(y=outer_low,  line=dict(color="white", width=1, dash="dot"),  annotation_text=f"Outer {outer_low:.2f}", annotation_position="left")
    figZ.add_hline(y=outer_high, line=dict(color="white", width=1, dash="dot"),  annotation_text=f"Outer {outer_high:.2f}", annotation_position="left")

    figZ.update_layout(
        title="Density Map — Zone Based",
        xaxis_title="Density",
        yaxis_title="MNQ Price",
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        font=dict(color="white"),
        bargap=0
    )
    st.plotly_chart(figZ, use_container_width=True)
else:
    st.info("Not enough data to compute density map.")

# =========================
# Modes
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
# Forecasting (defaults to 2 years lookback)
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
