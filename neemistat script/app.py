# Neemistat_App_Streamlit - Robust single-file Streamlit app
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt

# Try import statsmodels AutoReg; continue without it if missing
try:
    from statsmodels.tsa.ar_model import AutoReg
    STATS_INSTALLED = True
except Exception:
    STATS_INSTALLED = False

st.set_page_config(page_title="Neemistat", layout="wide")
st.title("Neemistat — Orderflow & Statistical Forecasting Made Simple")
st.markdown(
    "Upload intraday CSV (2 years OK). App detects basic columns and creates daily aggregates, "
    "histograms, orderflow metrics, quantile forecasts, and next-day True Range forecast (if statsmodels is installed)."
)

# Optional: clear cached function outputs if you want a fresh run during development
# st.cache_data.clear()

# ---------- Helpers ----------
@st.cache_data
def read_csv(file, datetime_col=None):
    # Read, trim column names, and attempt to detect/clean datetime column
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]

    # Try explicit datetime_col first
    if datetime_col and datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col].astype(str).str.strip(), errors='coerce')
    else:
        # try a few common names
        for cand in ["datetime", "timestamp", "time", "date"]:
            if cand in df.columns:
                df[cand] = pd.to_datetime(df[cand].astype(str).str.strip(), errors='coerce')
                datetime_col = cand
                break

    if datetime_col is None:
        raise ValueError("No datetime column found; upload a file with a datetime column or specify it.")
    # Drop rows where datetime couldn't be parsed
    df = df.dropna(subset=[datetime_col]).sort_values(datetime_col).reset_index(drop=True)
    return df, datetime_col

@st.cache_data
def to_daily_ohlc(df, datetime_col, price_cols):
    df = df.copy()

    # Ensure datetime column is real datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col].astype(str).str.strip(), errors='coerce')
    df = df.dropna(subset=[datetime_col])

    # Force numeric on chosen price columns BEFORE aggregation (some files have strings)
    for c in price_cols.values():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing any required price value
    df = df.dropna(subset=list(price_cols.values()))

    # Create date column for grouping (use date, not datetime to aggregate by day)
    df['date'] = df[datetime_col].dt.date

    agg = {
        price_cols['open']: 'first',
        price_cols['high']: 'max',
        price_cols['low']: 'min',
        price_cols['close']: 'last'
    }

    daily = df.groupby('date').agg(agg)
    # Convert index to datetime64 for plotting & consistency
    daily.index = pd.to_datetime(daily.index)
    daily.columns = ['open', 'high', 'low', 'close']

    # Ensure numeric after aggregation (safety)
    for col in ['open', 'high', 'low', 'close']:
        daily[col] = pd.to_numeric(daily[col], errors='coerce')

    # Drop days with any missing OHLC
    daily = daily.dropna(subset=['open','high','low','close'])

    return daily

def compute_orderflow(df, datetime_col):
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col].astype(str).str.strip(), errors='coerce')
    df = df.dropna(subset=[datetime_col])
    df['date'] = df[datetime_col].dt.date

    if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
        df['buy_volume'] = pd.to_numeric(df['buy_volume'], errors='coerce').fillna(0)
        df['sell_volume'] = pd.to_numeric(df['sell_volume'], errors='coerce').fillna(0)
        df['delta'] = df['buy_volume'] - df['sell_volume']
        daily_of = df.groupby('date').agg({'buy_volume':'sum','sell_volume':'sum','delta':'sum'})
    elif 'tick_direction' in df.columns and 'volume' in df.columns:
        df['tick_direction'] = pd.to_numeric(df['tick_direction'], errors='coerce').fillna(0)
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        df['delta'] = df['tick_direction'] * df['volume']
        daily_of = df.groupby('date').agg({'volume':'sum','delta':'sum'}).rename(columns={'volume':'total_volume'})
    else:
        return None

    daily_of.index = pd.to_datetime(daily_of.index)
    denom = daily_of.get('buy_volume', daily_of.get('total_volume', None))
    if denom is None:
        denom = (daily_of['delta'].abs() + 1e-9)  # fallback
    daily_of['imbalance'] = daily_of['delta'] / (denom + 1e-9)
    return daily_of

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

def true_range_ar1_forecast(daily, lookback=252):
    """Return next-day True Range % forecast using AR(1) on log(TR%)"""
    if not STATS_INSTALLED:
        return None
    df = daily.copy()
    # True Range % relative to price (use close)
    df['tr_pct'] = (df['high'] - df['low']) / df['close'] * 100
    series = df['tr_pct'].tail(lookback).dropna()
    if len(series) < 2:
        return None
    # log-transform for stability (avoid zeros)
    series_log = np.log(series.replace(0, 1e-9))
    model = AutoReg(series_log, lags=1, old_names=False)
    fit = model.fit()
    pred_log = fit.predict(start=len(series_log), end=len(series_log))
    pred_pct = float(np.exp(pred_log.iloc[0]))
    return pred_pct

# ---------- UI Inputs ----------
uploaded = st.file_uploader("Upload intraday CSV", type=['csv'])
if uploaded is None:
    st.info("Upload a CSV with at least datetime, open, high, low, close. Optional: buy_volume & sell_volume OR tick_direction & volume for orderflow.")
    st.stop()

# Read & clean
with st.spinner("Reading CSV..."):
    try:
        df_raw, dt_col = read_csv(uploaded)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

# Show columns & let user remap if needed
cols = df_raw.columns.tolist()
price_defaults = {
    'open': next((c for c in cols if c.lower().strip().startswith('open')), None),
    'high': next((c for c in cols if c.lower().strip().startswith('high')), None),
    'low': next((c for c in cols if c.lower().strip().startswith('low')), None),
    'close': next((c for c in cols if c.lower().strip().startswith('close')), None),
}
st.sidebar.header("Column mapping")
price_cols = {}
for k, v in price_defaults.items():
    # ensure index exists if v is None
    idx = cols.index(v) if (v in cols) else 0
    price_cols[k] = st.sidebar.selectbox(f"{k} column", options=[None] + cols, index=idx)

lookback = st.sidebar.number_input("Lookback (days) for quantile forecast", min_value=30, max_value=2000, value=500, step=10)
quantile_list = st.sidebar.multiselect("Quantiles to include", options=[0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99], default=[0.05,0.25,0.5,0.75,0.95])

# require mapping
if not all(price_cols.values()):
    st.error("Please map all price columns in the sidebar (open, high, low, close).")
    st.stop()

# ---------- Processing ----------
with st.spinner("Aggregating daily OHLC..."):
    daily = to_daily_ohlc(df_raw, dt_col, price_cols)

# Safety: ensure numeric (again)
for col in ['open','high','low','close']:
    daily[col] = pd.to_numeric(daily[col], errors='coerce')
daily = daily.dropna(subset=['open','high','low','close'])
if daily.empty:
    st.error("No valid daily OHLC rows after cleaning. Check your CSV.")
    st.stop()

# Bullish/bearish
daily['bullish'] = daily['close'] > daily['open']

# ---------- UI Outputs ----------
st.subheader("Daily OHLC — sample")
st.dataframe(daily.tail(10))

# HOD/LOD histograms
st.subheader("High / Low distributions (bullish vs bearish)")
fig_high = px.histogram(daily, x='high', color=daily['bullish'].map({True:'bullish', False:'bearish'}), nbins=50, title='High values by day type')
fig_low  = px.histogram(daily, x='low',  color=daily['bullish'].map({True:'bullish', False:'bearish'}), nbins=50, title='Low values by day type')
col1, col2 = st.columns(2)
with col1: st.plotly_chart(fig_high, use_container_width=True)
with col2: st.plotly_chart(fig_low, use_container_width=True)

# Orderflow if present
st.subheader("Orderflow (if present)")
daily_of = compute_orderflow(df_raw, dt_col)
if daily_of is None:
    st.info("No recognized orderflow columns found. Provide buy_volume & sell_volume OR tick_direction & volume to compute orderflow.")
else:
    st.dataframe(daily_of.tail(10))
    if 'delta' in daily_of.columns:
        st.line_chart(daily_of['delta'])
    if 'imbalance' in daily_of.columns:
        st.line_chart(daily_of['imbalance'])

# Quantile forecast
st.subheader("Quantile-based Forecast (next day)")
if len(quantile_list) == 0:
    st.warning("Select at least one quantile in the sidebar to compute the forecast.")
else:
    with st.spinner("Computing quantile forecast..."):
        qdf = quantile_forecast(daily, lookback=lookback, quantiles=sorted(quantile_list))
    st.dataframe(qdf)
    fig_q = px.line(qdf, x='quantile', y=['forecast_high','forecast_close','forecast_low'], markers=True)
    fig_q.update_layout(title='Forecast bands vs quantile (applied to last open)')
    st.plotly_chart(fig_q, use_container_width=True)

# True Range forecast
st.subheader("Next-day True Range Forecast (%)")
if not STATS_INSTALLED:
    st.info("statsmodels not installed — True Range AR(1) forecast is disabled. Install statsmodels to enable it.")
    tr_forecast = None
else:
    with st.spinner("Computing True Range AR(1) forecast..."):
        tr_forecast = true_range_ar1_forecast(daily, lookback=252)
    if tr_forecast is None:
        st.info("Not enough historical True Range data to forecast.")
    else:
        st.metric("Predicted True Range %", f"{tr_forecast:.3f}%")

# Interactive daily OHLC chart (as lines)
st.subheader("Interactive daily price chart")
daily_plot = daily.reset_index().rename(columns={'index':'date'})
daily_plot['date'] = pd.to_datetime(daily_plot['date'])
fig_candle = px.line(daily_plot, x='date', y=['open','high','low','close'], title='Daily OHLC (lines)')
st.plotly_chart(fig_candle, use_container_width=True)

# Downloadable report (CSV + hist images)
st.subheader("Download report")
report_name = st.text_input("Report base name", value="Neemistat_report")
make_report = st.button("Generate & Download ZIP")
if make_report:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        # quantile forecast csv
        if 'qdf' in locals() and not qdf.empty:
            zf.writestr(f"{report_name}_quantile_forecast.csv", qdf.to_csv(index=False).encode())
        # save histograms
        for colname in ['high','low']:
            fig, ax = plt.subplots()
            daily[colname].plot(kind='hist', bins=50, ax=ax)
            ax.set_title(f"{colname.capitalize()} distribution")
            imgbuf = io.BytesIO()
            plt.savefig(imgbuf, format='png')
            plt.close(fig)
            zf.writestr(f"{report_name}_{colname}_hist.png", imgbuf.getvalue())
        # daily sample
        zf.writestr(f"{report_name}_daily_sample.csv", daily.tail(50).to_csv().encode())
        # orderflow
        if daily_of is not None:
            zf.writestr(f"{report_name}_orderflow.csv", daily_of.to_csv().encode())
    buf.seek(0)
    st.download_button(label="Download Neemistat report (ZIP)", data=buf, file_name=f"{report_name}.zip", mime="application/zip")

st.markdown("---")
st.caption("Neemistat — lightweight Streamlit rebrand of QuantiStat. True Range forecasting uses AR(1) (requires statsmodels).")
