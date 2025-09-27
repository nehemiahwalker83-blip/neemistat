# app.py - Neemistat+ (full feature set)
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time as t

# statsmodels optional
try:
    from statsmodels.tsa.ar_model import AutoReg
    STATS_INSTALLED = True
except Exception:
    STATS_INSTALLED = False

st.set_page_config(page_title="Neemistat+", layout="wide")
st.title("Neemistat+ — Density, HOD/LOD, Forecasting, True Range")
st.markdown("Upload intraday CSV. Required columns: datetime, open, high, low, close. Optional: volume, tick_direction, buy_volume, sell_volume.")

# --------------------
# Helpers & processing
# --------------------
@st.cache_data
def read_and_clean(file, datetime_hint=None):
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]
    # pick datetime column
    dt_col = None
    if datetime_hint and datetime_hint in df.columns:
        dt_col = datetime_hint
    else:
        for cand in ["datetime","timestamp","time","date","ts"]:
            if cand in df.columns:
                dt_col = cand
                break
    if dt_col is None:
        raise ValueError("No datetime-like column found.")
    df[dt_col] = pd.to_datetime(df[dt_col].astype(str).str.strip(), errors='coerce')
    df = df.dropna(subset=[dt_col]).reset_index(drop=True)
    # strip whitespace in object columns
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].str.strip()
    return df, dt_col

@st.cache_data
def make_daily(df, dt_col, price_cols):
    df = df.copy()
    # ensure numeric
    for c in price_cols.values():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=list(price_cols.values()))
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
    for c in ['open','high','low','close']:
        daily[c] = pd.to_numeric(daily[c], errors='coerce')
    daily = daily.dropna(subset=['open','high','low','close'])
    return daily

def density_map_price_based(df, dt_col, price_cols, bins_time=200, bins_price=150):
    tmp = df.copy()
    # choose price column (close preferred)
    price_col = price_cols.get('close') if price_cols.get('close') in tmp.columns else price_cols.get('open')
    tmp['tod'] = tmp[dt_col].dt.hour*3600 + tmp[dt_col].dt.minute*60 + tmp[dt_col].dt.second
    tmp['date_only'] = tmp[dt_col].dt.date
    opens = tmp.groupby('date_only')[price_cols['open']].first().rename('day_open')
    tmp = tmp.merge(opens, left_on='date_only', right_index=True, how='left')
    tmp['price_offset'] = tmp[price_col] - tmp['day_open']
    x = tmp['tod'].values
    y = tmp['price_offset'].values
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]; y = y[mask]
    if len(x) == 0:
        return None, None, None
    H, xedges, yedges = np.histogram2d(x, y, bins=[bins_time, bins_price])
    return H.T, xedges, yedges

def compute_hod_lod_by_bull_bear(daily):
    df = daily.copy()
    df['bullish'] = df['close'] > df['open']
    bullish = df[df['bullish']]
    bearish = df[~df['bullish']]
    return {
        'bullish_highs': bullish['high'].values,
        'bullish_lows': bullish['low'].values,
        'bearish_highs': bearish['high'].values,
        'bearish_lows': bearish['low'].values
    }

def quantile_forecast(daily, lookback=500, quantiles=[0.05,0.25,0.5,0.75,0.95]):
    df = daily.copy()
    df['high_off_open'] = df['high'] - df['open']
    df['low_off_open'] = df['low'] - df['open']
    df['close_off_open'] = df['close'] - df['open']
    last_open = df['open'].iloc[-1]
    q = df[['high_off_open','low_off_open','close_off_open']].tail(lookback).quantile(quantiles)
    rows = []
    for name in q.index:
        rows.append({
            'quantile': float(name),
            'forecast_high': float(last_open + q.loc[name,'high_off_open']),
            'forecast_low': float(last_open + q.loc[name,'low_off_open']),
            'forecast_close': float(last_open + q.loc[name,'close_off_open'])
        })
    return pd.DataFrame(rows)

def true_range_ar1_with_ci(daily, lookback=504, alpha=0.05):
    """
    Fit AR(1) on log(TR%) and return forecast pct plus (lower, upper) for given alpha.
    Returns (pred_pct, lower_pct, upper_pct) or (None,None,None) if not available.
    """
    if not STATS_INSTALLED:
        return None, None, None
    df = daily.copy()
    df['tr_pct'] = (df['high'] - df['low']) / df['close'] * 100
    s = df['tr_pct'].tail(lookback).dropna()
    if len(s) < 3:
        return None, None, None
    s_log = np.log(s.replace(0, 1e-9))
    model = AutoReg(s_log, lags=1, old_names=False)
    res = model.fit()
    pred_log = res.predict(start=len(s_log), end=len(s_log))
    pred_log_val = float(pred_log.iloc[0])
    # residual variance (sigma2)
    sigma2 = float(getattr(res, 'sigma2', np.var(res.resid, ddof=1)))
    # one-step forecast variance for AR(1) is sigma2 (approx)
    se = np.sqrt(sigma2)
    z = 1.96  # ~95%
    lower_log = pred_log_val - z*se
    upper_log = pred_log_val + z*se
    pred_pct = float(np.exp(pred_log_val))
    lower_pct = float(np.exp(lower_log))
    upper_pct = float(np.exp(upper_log))
    return pred_pct, lower_pct, upper_pct

# --------------------
# Sidebar & controls
# --------------------
st.sidebar.header("Options")
live = st.sidebar.checkbox("Live updates (re-run every 2s)", value=False)
bias_mode = st.sidebar.radio("Bias selection", options=["Prediction","Forecasting"])
mode = st.sidebar.selectbox("Mode", options=["Price-Based","Time-Based","Level Validator"])
lookback_years = st.sidebar.number_input("Historical years for forecasting", min_value=1, max_value=5, value=2)
lookback_days = int(252 * lookback_years)

uploaded = st.file_uploader("Upload intraday CSV", type=['csv'])
if uploaded is None:
    st.info("Upload intraday CSV with datetime, open, high, low, close.")
    st.stop()

# Read + column mapping UI
raw_df, dt_guess = read_and_clean(uploaded)
cols = raw_df.columns.tolist()
st.sidebar.markdown("### Column mapping")
open_col = st.sidebar.selectbox("open column", options=cols, index=cols.index(next((c for c in cols if c.lower().startswith('open')), cols[0])))
high_col = st.sidebar.selectbox("high column", options=cols, index=cols.index(next((c for c in cols if c.lower().startswith('high')), cols[0])))
low_col = st.sidebar.selectbox("low column", options=cols, index=cols.index(next((c for c in cols if c.lower().startswith('low')), cols[0])))
close_col = st.sidebar.selectbox("close column", options=cols, index=cols.index(next((c for c in cols if c.lower().startswith('close')), cols[0])))
dt_col = st.sidebar.selectbox("datetime column", options=cols, index=cols.index(dt_guess))

price_cols = {'open':open_col, 'high':high_col, 'low':low_col, 'close':close_col}

# Level validator input
levels_input = st.sidebar.text_input("Levels (comma separated) for Level Validator", value="")
levels = []
if mode == "Level Validator" and levels_input.strip():
    try:
        levels = [float(x.strip()) for x in levels_input.split(",") if x.strip()!='']
    except Exception:
        st.sidebar.error("Levels must be numeric, comma-separated.")

# --------------------
# Aggregations
# --------------------
with st.spinner("Aggregating daily OHLC..."):
    daily = make_daily(raw_df, dt_col, price_cols)

# Safety checks
for c in ['open','high','low','close']:
    daily[c] = pd.to_numeric(daily[c], errors='coerce')
daily = daily.dropna(subset=['open','high','low','close'])
if daily.empty:
    st.error("No valid daily OHLC rows after cleaning. Check mapping or CSV format.")
    st.stop()

st.subheader("Daily OHLC — sample")
st.dataframe(daily.tail(10))

# HOD/LOD split
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

# Density map
st.subheader("Density map")
if mode in ["Price-Based","Time-Based"]:
    H, xedges, yedges = density_map_price_based(raw_df, dt_col, price_cols, bins_time=200, bins_price=150)
    if H is None:
        st.info("Not enough intraday rows for density map.")
    else:
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        df_tmp = raw_df.copy()
        df_tmp['date_only'] = df_tmp[dt_col].dt.date
        opens = df_tmp.groupby('date_only')[price_cols['open']].first().rename('day_open')
        df_tmp = df_tmp.merge(opens, left_on='date_only', right_index=True, how='left')
        df_tmp['price_offset'] = df_tmp[price_cols['close']] - df_tmp['day_open']
        p25, p50, p75 = np.nanpercentile(df_tmp['price_offset'].dropna(), [25,50,75])
        heat = go.Figure(go.Heatmap(z=H, x=xcenters, y=ycenters, colorscale='Viridis'))
        heat.add_hline(y=p50, line=dict(color='white', width=2))
        heat.add_hline(y=p25, line=dict(color='white', width=1, dash='dash'))
        heat.add_hline(y=p75, line=dict(color='white', width=1, dash='dash'))
        if mode == "Time-Based":
            # convert x ticks to hh:mm labels
            xticks = [int(x) for x in np.linspace(xcenters.min(), xcenters.max(), 6)]
            xticklabels = []
            for s in xticks:
                hh = int(s//3600); mm = int((s%3600)//60)
                xticklabels.append(f"{hh:02d}:{mm:02d}")
            heat.update_layout(xaxis=dict(tickmode='array', tickvals=xticks, ticktext=xticklabels))
        heat.update_layout(title="Density map (time vs price offset)")
        st.plotly_chart(heat, use_container_width=True)
else:
    # Level validator
    st.subheader("Level Validator")
    if not levels:
        st.info("Enter levels in the sidebar to compute historical hit rates.")
    else:
        recent = daily.tail(lookback_days)
        rows = []
        for lvl in levels:
            touched = ((recent['low'] <= lvl) & (recent['high'] >= lvl)).sum()
            pct = touched / len(recent) * 100.0 if len(recent)>0 else np.nan
            rows.append({'level': lvl, 'hit_rate_%': pct})
        st.dataframe(pd.DataFrame(rows))

# Next-day True Range (pct & points) + TR plot with forecast & CI
st.subheader("Next-day True Range")
if STATS_INSTALLED:
    pred_pct, lower_pct, upper_pct = true_range_ar1_with_ci(daily, lookback=lookback_days, alpha=0.05)
    if pred_pct is None:
        st.info("Not enough history for AR(1). Using persistence.")
        pred_pct = (daily['high'] - daily['low']).div(daily['close']).mul(100).iloc[-1]
        lower_pct = upper_pct = None
else:
    pred_pct = (daily['high'] - daily['low']).div(daily['close']).mul(100).iloc[-1]
    lower_pct = upper_pct = None

last_close = float(daily['close'].iloc[-1])
pred_points = last_close * (pred_pct/100.0)
colp, colq = st.columns(2)
colp.metric("TR (percent)", f"{pred_pct:.3f}%")
colq.metric("TR (points)", f"{pred_points:.4f}")

# TR historical plot with forecast point & CI
tr_hist = (daily['high'] - daily['low'])/daily['close']*100
fig_tr = go.Figure()
fig_tr.add_trace(go.Scatter(x=tr_hist.index, y=tr_hist.values, mode='lines', name='Historical TR%'))
if pred_pct is not None:
    # put forecast as next-day point at date = last_date + 1 day
    next_date = tr_hist.index[-1] + pd.Timedelta(days=1)
    fig_tr.add_trace(go.Scatter(x=[next_date], y=[pred_pct], mode='markers', name='AR1 forecast', marker=dict(size=10, color='red')))
    if lower_pct is not None and upper_pct is not None:
        fig_tr.add_trace(go.Scatter(x=[next_date,next_date], y=[lower_pct, upper_pct], mode='lines+markers', name='95% CI', marker=dict(size=6, color='orange')))
fig_tr.update_layout(title='True Range % history with next-day forecast', yaxis_title='TR %')
st.plotly_chart(fig_tr, use_container_width=True)

# Quantile forecast
st.subheader("Quantile forecast")
qdf = quantile_forecast(daily, lookback=lookback_days, quantiles=[0.05,0.25,0.5,0.75,0.95])
if qdf is None or qdf.empty:
    st.info("Insufficient history for quantile forecast.")
else:
    st.dataframe(qdf)
    figq = px.line(qdf, x='quantile', y=['forecast_high','forecast_close','forecast_low'], markers=True)
    figq.update_layout(title='Quantile bands (applied to last open)')
    st.plotly_chart(figq, use_container_width=True)

# Candlestick (Plotly) chart for daily OHLC
st.subheader("Candlestick / OHLC chart")
daily_plot = daily.reset_index().rename(columns={'index':'date'})
# candlestick with plotly.graph_objects
fig_candle = go.Figure(data=[go.Candlestick(x=daily_plot['date'],
                open=daily_plot['open'], high=daily_plot['high'],
                low=daily_plot['low'], close=daily_plot['close'],
                name='OHLC')])
fig_candle.update_layout(title='Daily Candlestick', xaxis_title='Date')
st.plotly_chart(fig_candle, use_container_width=True)

# Downloadable report
st.subheader("Download report")
report_name = st.text_input("Report name", value="Neemistat_report")
if st.button("Generate ZIP report"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        if not qdf.empty:
            zf.writestr(f"{report_name}_quantile.csv", qdf.to_csv(index=False).encode())
        zf.writestr(f"{report_name}_daily_sample.csv", daily.tail(50).to_csv().encode())
        for cname in ['high','low']:
            fig, ax = plt.subplots()
            daily[cname].plot(kind='hist', bins=50, ax=ax)
            ax.set_title(f"{cname.capitalize()} distribution")
            imgb = io.BytesIO()
            plt.savefig(imgb, format='png')
            plt.close(fig)
            zf.writestr(f"{report_name}_{cname}_hist.png", imgb.getvalue())
        try:
            df_of = compute_orderflow(raw_df, dt_col)
            if df_of is not None:
                zf.writestr(f"{report_name}_orderflow.csv", df_of.to_csv().encode())
        except Exception:
            pass
    buf.seek(0)
    st.download_button("Download ZIP", data=buf, file_name=f"{report_name}.zip", mime="application/zip")

# Live updates (re-run every 2s)
if live:
    t.sleep(2)
    st.experimental_rerun()

st.markdown("---")
st.caption("Neemistat+ — density, HOD/LOD split, quantile forecasting, AR(1) True Range, candlestick chart.")
