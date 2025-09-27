# app.py - Neemistat extended
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
from datetime import datetime, time
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time as t

# Try import statsmodels AutoReg for AR(1) true-range forecasting
try:
    from statsmodels.tsa.ar_model import AutoReg
    STATS_INSTALLED = True
except Exception:
    STATS_INSTALLED = False

st.set_page_config(page_title="Neemistat+", layout="wide")
st.title("Neemistat+ — Density, HOD/LOD, Forecasting, True Range")
st.markdown("Upload intraday CSV. Required columns: datetime, open, high, low, close. Optional: volume, tick_direction, buy_volume, sell_volume.")

# ---------- Helpers ----------
@st.cache_data
def read_and_clean(file, datetime_col_hint=None):
    df = pd.read_csv(file)
    # normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    # detect datetime column
    dt_col = None
    if datetime_col_hint and datetime_col_hint in df.columns:
        dt_col = datetime_col_hint
    else:
        for cand in ["datetime","timestamp","time","date","ts"]:
            if cand in df.columns:
                dt_col = cand
                break
    if dt_col is None:
        raise ValueError("No datetime-like column found. Please include a datetime column.")
    # convert datetime
    df[dt_col] = pd.to_datetime(df[dt_col].astype(str).str.strip(), errors='coerce')
    df = df.dropna(subset=[dt_col])
    # strip whitespace in string columns
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].str.strip()
    return df, dt_col

@st.cache_data
def make_daily(df, dt_col, price_cols):
    df = df.copy()
    # ensure numeric for chosen price columns
    for c in price_cols.values():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # drop rows missing required price columns
    df = df.dropna(subset=list(price_cols.values()))
    # create date
    df['date'] = df[dt_col].dt.date
    agg = {
        price_cols['open']: 'first',
        price_cols['high']: 'max',
        price_cols['low']: 'min',
        price_cols['close']: 'last'
    }
    daily = df.groupby('date').agg(agg)
    daily.index = pd.to_datetime(daily.index)
    daily.columns = ['open','high','low','close']
    # ensure numeric
    for c in ['open','high','low','close']:
        daily[c] = pd.to_numeric(daily[c], errors='coerce')
    daily = daily.dropna(subset=['open','high','low','close'])
    return daily

def compute_true_range_forecast_AR1(daily, lookback_days=504):
    """Return next-day true range percent (as percentage) using AR(1) on log(TR%)."""
    if not STATS_INSTALLED:
        return None
    df = daily.copy()
    df['tr_pct'] = (df['high'] - df['low']) / df['close'] * 100
    s = df['tr_pct'].tail(lookback_days).dropna()
    if len(s) < 3:
        return None
    s_log = np.log(s.replace(0, 1e-9))
    model = AutoReg(s_log, lags=1, old_names=False)
    fit = model.fit()
    pred_log = fit.predict(start=len(s_log), end=len(s_log))
    pred_pct = float(np.exp(pred_log.iloc[0]))
    return pred_pct

def compute_true_range_persistence(daily):
    """Simple persistence: next TR% = last TR%"""
    df = daily.copy()
    df['tr_pct'] = (df['high'] - df['low']) / df['close'] * 100
    last = df['tr_pct'].dropna()
    return float(last.iloc[-1]) if len(last)>0 else None

def density_map_price_based(df, dt_col, price_cols, bins_time=100, bins_price=200):
    """
    Build a 2D histogram (heatmap) of price offset vs time-of-day
    x: minute-of-day (or fractional day)
    y: price offset = price - open (or price ratio)
    """
    tmp = df.copy()
    # require intraday rows: use price column (close if not present)
    price_col = price_cols.get('close', None)
    if price_col not in tmp.columns:
        price_col = price_cols.get('open')
    # create time of day in seconds fraction
    tmp['tod'] = tmp[dt_col].dt.hour*3600 + tmp[dt_col].dt.minute*60 + tmp[dt_col].dt.second
    # price offset relative to open for each day — need mapping from date to that day's open
    tmp['date_only'] = tmp[dt_col].dt.date
    opens = tmp.groupby('date_only')[price_cols['open']].first().rename('day_open')
    tmp = tmp.merge(opens, left_on='date_only', right_index=True, how='left')
    tmp['price_offset'] = tmp[price_col] - tmp['day_open']  # in points
    # build 2d histogram
    x = tmp['tod'].values
    y = tmp['price_offset'].values
    # remove nans
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]; y = y[mask]
    if len(x)==0:
        return None, None, None
    H, xedges, yedges = np.histogram2d(x, y, bins=[bins_time, bins_price])
    return H.T, xedges, yedges  # transpose so y rows, x cols for heatmap plotting

def compute_hod_lod_by_bull_bear(daily):
    df = daily.copy()
    df['bullish'] = df['close'] > df['open']
    bullish = df[df['bullish']]
    bearish = df[~df['bullish']]
    # HOD and LOD arrays
    res = {
        'bullish_highs': bullish['high'].values,
        'bullish_lows': bullish['low'].values,
        'bearish_highs': bearish['high'].values,
        'bearish_lows': bearish['low'].values
    }
    return res

def level_validator_hit_rate(df_raw, dt_col, price_cols, levels, lookback_days=504):
    """
    For each level, compute historical percent of days where intraday price touched that level.
    We'll look at last `lookback_days` of daily data and see intraday high/low touches.
    """
    # Make daily intraday highs/lows already computed.
    daily = make_daily(df_raw, dt_col, price_cols)
    # For last lookback_days:
    recent = daily.tail(lookback_days)
    results = {}
    for lvl in levels:
        touched = ((recent['low'] <= lvl) & (recent['high'] >= lvl)).sum()
        results[lvl] = touched / len(recent) * 100.0 if len(recent)>0 else np.nan
    return results

# ---------- Sidebar / Controls ----------
st.sidebar.header("Options")
live = st.sidebar.checkbox("Live updates (re-run every 2s)", value=False)
bias_mode = st.sidebar.radio("Bias selection", options=["Prediction","Forecasting"])
mode = st.sidebar.selectbox("Mode", options=["Price-Based","Time-Based","Level Validator"])
lookback_years = st.sidebar.number_input("Historical years for forecasting", min_value=1, max_value=5, value=2)
lookback_days_2yrs = int(252 * lookback_years)

# upload
uploaded = st.file_uploader("Upload intraday CSV", type=['csv'])
if uploaded is None:
    st.info("Upload intraday CSV with datetime, open, high, low, close columns.")
    st.stop()

# column mapping UI
raw_df, dt_guess = read_and_clean(uploaded)
cols = raw_df.columns.tolist()
st.sidebar.markdown("### Column mapping")
open_col = st.sidebar.selectbox("open column", options=cols, index=cols.index(next((c for c in cols if c.lower().startswith('open')), cols[0])))
high_col = st.sidebar.selectbox("high column", options=cols, index=cols.index(next((c for c in cols if c.lower().startswith('high')), cols[0])))
low_col = st.sidebar.selectbox("low column", options=cols, index=cols.index(next((c for c in cols if c.lower().startswith('low')), cols[0])))
close_col = st.sidebar.selectbox("close column", options=cols, index=cols.index(next((c for c in cols if c.lower().startswith('close')), cols[0])))
datetime_col = st.sidebar.selectbox("datetime column", options=cols, index=cols.index(dt_guess))

price_cols = {'open':open_col, 'high':high_col, 'low':low_col, 'close':close_col}
dt_col = datetime_col

# optional level validator inputs
levels_input = st.sidebar.text_input("Levels (comma separated) for Level Validator", value="")
levels = []
if mode == "Level Validator" and levels_input.strip():
    try:
        levels = [float(x.strip()) for x in levels_input.split(",") if x.strip()!='']
    except Exception:
        st.sidebar.error("Levels must be numeric, comma-separated.")

# ---------- Compute daily and other aggregates ----------
with st.spinner("Aggregating daily OHLC..."):
    daily = make_daily(raw_df, dt_col, price_cols)

# Safety: ensure numeric types
for c in ['open','high','low','close']:
    daily[c] = pd.to_numeric(daily[c], errors='coerce')
daily = daily.dropna(subset=['open','high','low','close'])
if daily.empty:
    st.error("No daily OHLC rows after cleaning. Check mapping / CSV format.")
    st.stop()

# show sample
st.subheader("Daily OHLC sample")
st.dataframe(daily.tail(10))

# HOD/LOD by bullish/bearish
hodlod = compute_hod_lod_by_bull_bear(daily)
st.subheader("HOD / LOD (bullish vs bearish)")
colA, colB = st.columns(2)
with colA:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=hodlod['bullish_highs'], name='Bullish HOD', opacity=0.7, nbinsx=50))
    fig.add_trace(go.Histogram(x=hodlod['bearish_highs'], name='Bearish HOD', opacity=0.6, nbinsx=50))
    fig.update_layout(barmode='overlay', title="High of Day distribution (bullish vs bearish)")
    st.plotly_chart(fig, use_container_width=True)
with colB:
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=hodlod['bullish_lows'], name='Bullish LOD', opacity=0.7, nbinsx=50))
    fig2.add_trace(go.Histogram(x=hodlod['bearish_lows'], name='Bearish LOD', opacity=0.6, nbinsx=50))
    fig2.update_layout(barmode='overlay', title="Low of Day distribution (bullish vs bearish)")
    st.plotly_chart(fig2, use_container_width=True)

# Density map area
st.subheader("Density map")
if mode == "Price-Based" or mode == "Time-Based":
    H, xedges, yedges = density_map_price_based(raw_df, dt_col, price_cols, bins_time=200, bins_price=150)
    if H is None:
        st.info("Not enough intraday rows for density map.")
    else:
        # build heatmap
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        # convert seconds -> hh:mm for ticks if time-based
        if mode == "Time-Based":
            xvals = xcenters
            xticks = [int(x) for x in np.linspace(xvals.min(), xvals.max(), 6)]
            xticklabels = []
            for s in xticks:
                hh = int(s // 3600); mm = int((s % 3600)//60)
                xticklabels.append(f"{hh:02d}:{mm:02d}")
            fig = go.Figure(data=go.Heatmap(z=H, x=xcenters, y=ycenters, colorscale='Viridis'))
            # add median/25/75 lines across y using percentiles of price offset
            price_offsets = (ycenters)
            # compute percentiles across all daily price offsets
            df_tmp = raw_df.copy()
            df_tmp['date_only'] = df_tmp[dt_col].dt.date
            opens = df_tmp.groupby('date_only')[price_cols['open']].first().rename('day_open')
            df_tmp = df_tmp.merge(opens, left_on='date_only', right_index=True, how='left')
            df_tmp['price_offset'] = df_tmp[price_cols['close']] - df_tmp['day_open']
            p25, p50, p75 = np.nanpercentile(df_tmp['price_offset'].dropna(), [25,50,75])
            fig.add_hline(y=p50, line=dict(color='white', width=2), annotation_text='median', annotation_position='bottom right')
            fig.add_hline(y=p25, line=dict(color='white', width=1, dash='dash'))
            fig.add_hline(y=p75, line=dict(color='white', width=1, dash='dash'))
            fig.update_layout(title="Density map (time vs price offset)", xaxis=dict(tickmode='array', tickvals=xticks, ticktext=xticklabels))
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Price-based: x axis is time seconds, y is price offset (points)
            fig = go.Figure(data=go.Heatmap(z=H, x=xcenters, y=ycenters, colorscale='Viridis'))
            # overlay median/25/75 as before
            df_tmp = raw_df.copy()
            df_tmp['date_only'] = df_tmp[dt_col].dt.date
            opens = df_tmp.groupby('date_only')[price_cols['open']].first().rename('day_open')
            df_tmp = df_tmp.merge(opens, left_on='date_only', right_index=True, how='left')
            df_tmp['price_offset'] = df_tmp[price_cols['close']] - df_tmp['day_open']
            p25, p50, p75 = np.nanpercentile(df_tmp['price_offset'].dropna(), [25,50,75])
            fig.add_hline(y=p50, line=dict(color='white', width=2), annotation_text='median', annotation_position='bottom right')
            fig.add_hline(y=p25, line=dict(color='white', width=1, dash='dash'))
            fig.add_hline(y=p75, line=dict(color='white', width=1, dash='dash'))
            fig.update_layout(title="Density map (time vs price offset in points)")
            st.plotly_chart(fig, use_container_width=True)

elif mode == "Level Validator":
    st.info("Level Validator mode: shows historical hit rates for user-provided levels.")
    if not levels:
        st.info("Enter levels in the sidebar (comma-separated) to compute hit rates.")
    else:
        hits = level_validator_hit_rate(raw_df, dt_col, price_cols, levels, lookback_days=lookback_days_2yrs)
        df_hits = pd.DataFrame({'level': list(hits.keys()), 'hit_rate_%': list(hits.values())})
        st.dataframe(df_hits)

# Next-day True Range forecast (point & percent)
st.subheader("Next-day True Range forecast")
if STATS_INSTALLED:
    tr_pct = true_range_ar1_forecast(daily, lookback_days_2yrs)
    if tr_pct is None:
        st.info("Not enough data for AR(1) TR forecast. Falling back to persistence.")
        tr_pct = compute_true_range_persistence(daily)
else:
    tr_pct = compute_true_range_persistence(daily)

if tr_pct is None:
    st.info("Not enough True Range history to produce forecast.")
else:
    # in percent; also compute points using last close
    last_close = float(daily['close'].iloc[-1])
    tr_points = last_close * (tr_pct/100.0)
    col1, col2 = st.columns(2)
    col1.metric("TR (percent)", f"{tr_pct:.3f}%")
    col2.metric("TR (points)", f"{tr_points:.4f}")

# Forecasting using 2 years (quantile bands)
st.subheader("Quantile forecast (2-year lookback by default)")
with st.spinner("Computing quantile forecast..."):
    qdf = quantile_forecast(daily, lookback=lookback_days_2yrs, quantiles=[0.05,0.25,0.5,0.75,0.95])
if qdf is not None and not qdf.empty:
    st.dataframe(qdf)
    fig_q = px.line(qdf, x='quantile', y=['forecast_high','forecast_close','forecast_low'], markers=True)
    fig_q.update_layout(title="Quantile bands (applied to last open)")
    st.plotly_chart(fig_q, use_container_width=True)
else:
    st.info("Quantile forecast not available (insufficient history).")

# Downloadable ZIP report
st.subheader("Download report")
report_name = st.text_input("Report name", value="Neemistat_report")
if st.button("Generate ZIP report"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        # quantile CSV
        if 'qdf' in locals() and qdf is not None:
            zf.writestr(f"{report_name}_quantile.csv", qdf.to_csv(index=False).encode())
        # daily sample
        zf.writestr(f"{report_name}_daily_sample.csv", daily.tail(50).to_csv().encode())
        # hod/lod histograms
        for cname in ['high','low']:
            fig, ax = plt.subplots()
            daily[cname].plot(kind='hist', bins=50, ax=ax)
            ax.set_title(f"{cname.capitalize()} distribution")
            imgb = io.BytesIO()
            plt.savefig(imgb, format='png')
            plt.close(fig)
            zf.writestr(f"{report_name}_{cname}_hist.png", imgb.getvalue())
        # orderflow if present
        try:
            daily_of = compute_orderflow(raw_df, dt_col)
            if daily_of is not None:
                zf.writestr(f"{report_name}_orderflow.csv", daily_of.to_csv().encode())
        except Exception:
            pass
    buf.seek(0)
    st.download_button("Download ZIP", data=buf, file_name=f"{report_name}.zip", mime="application/zip")

st.markdown("---")
st.caption("Neemistat+ — density map, HOD/LOD, forecasting, True Range. Live update option available.")

# ---------- Live updates ----------
if live:
    # pause then re-run. This pattern causes Streamlit to rerun the script every 2s.
    t.sleep(2)
    st.experimental_rerun()
