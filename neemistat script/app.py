# Neemistat_App_Streamlit - Streamlit single-file app
# Features:
# - Upload intraday CSV (2 years OK)
# - Automatic OHLC detection and daily aggregation
# - Bullish/bearish day calculation
# - HOD/LOD histograms
# - Orderflow metrics if columns exist
# - Quantile-based forecast
# - True Range AR(1) forecast for next day
# - Interactive charts
# - Downloadable report (CSV + histograms)

import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

st.set_page_config(page_title="Neemistat", layout="wide")
st.title("Neemistat — Orderflow & Statistical Forecasting Made Simple")
st.markdown("Upload intraday CSV (2 years OK). The app detects basic columns, creates daily aggregates, histograms, orderflow metrics, quantile forecasts, and True Range forecasts.")

# ---------- Helpers ----------
@st.cache_data
def read_csv(file, datetime_col=None):
    df = pd.read_csv(file)
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    if datetime_col and datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col].astype(str).str.strip(), errors='coerce')
    else:
        # try common names
        for cand in ["datetime","timestamp","time","date"]:
            if cand in df.columns:
                df[cand] = pd.to_datetime(df[cand].astype(str).str.strip(), errors='coerce')
                datetime_col = cand
                break
    if datetime_col is None:
        raise ValueError("No datetime column found; please upload a file with a datetime column or specify it.")
    df = df.dropna(subset=[datetime_col]).sort_values(datetime_col).reset_index(drop=True)
    return df, datetime_col

@st.cache_data
def to_daily_ohlc(df, datetime_col, price_cols):
    df = df.copy()
    # Ensure numeric OHLC
    for col in price_cols.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=list(price_cols.values()))
    
    # Create 'date'
    df['date'] = df[datetime_col].dt.date
    agg = {
        price_cols['open']: 'first',
        price_cols['high']: 'max',
        price_cols['low']: 'min',
        price_cols['close']: 'last'
    }
    daily = df.groupby('date').agg(agg)
    daily.index = pd.to_datetime(daily.index)
    daily.columns = ['open','high','low','close']
    return daily

def compute_orderflow(df, datetime_col):
    df = df.copy()
    df['date'] = pd.to_datetime(df[datetime_col].dt.date)
    if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
        df['delta'] = df['buy_volume'] - df['sell_volume']
        daily_of = df.groupby('date').agg({'buy_volume':'sum','sell_volume':'sum','delta':'sum'})
    elif 'tick_direction' in df.columns and 'volume' in df.columns:
        df['delta'] = df['tick_direction'] * df['volume']
        daily_of = df.groupby('date').agg({'volume':'sum','delta':'sum'})
        daily_of = daily_of.rename(columns={'volume':'total_volume'})
    else:
        return None
    daily_of['imbalance'] = daily_of['delta'] / (daily_of.get('buy_volume', daily_of.get('total_volume', 1)) + 1e-9)
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

def true_range_forecast(daily):
    """Forecast next day True Range % using AR(1) on log TR%"""
    df = daily.copy()
    # True Range % = (high - low)/open
    df['tr_pct'] = (df['high'] - df['low']) / df['open']
    df['log_tr'] = np.log(df['tr_pct'].replace(0, 1e-9))  # avoid log(0)
    # Fit AR(1)
    model = AutoReg(df['log_tr'], lags=1, old_names=False)
    fit = model.fit()
    # Predict next day
    pred_log = fit.predict(start=len(df), end=len(df))
    tr_forecast = np.exp(pred_log.iloc[0])
    return tr_forecast

# ---------- UI Inputs ----------
uploaded = st.file_uploader("Upload intraday CSV", type=['csv'])
if uploaded is None:
    st.info("Upload a CSV with at least datetime, open, high, low, close. Optional: buy_volume & sell_volume OR tick_direction & volume for orderflow.")
    st.stop()

with st.spinner('Reading file...'):
    try:
        df_raw, dt_col = read_csv(uploaded)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

# Detect price columns
cols = df_raw.columns.tolist()
price_defaults = {
    'open': next((c for c in cols if c.lower().startswith('open')), None),
    'high': next((c for c in cols if c.lower().startswith('high')), None),
    'low': next((c for c in cols if c.lower().startswith('low')), None),
    'close': next((c for c in cols if c.lower().startswith('close')), None),
}
st.sidebar.header('Column mapping')
price_cols = {}
for k,v in price_defaults.items():
    price_cols[k] = st.sidebar.selectbox(f"{k} column", options=[None]+cols, index=cols.index(v) if v in cols else 0)

lookback = st.sidebar.number_input('Lookback (days) for quantile forecast', min_value=30, max_value=2000, value=500, step=10)
quantile_list = st.sidebar.multiselect('Quantiles to include', options=[0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99], default=[0.05,0.25,0.5,0.75,0.95])

if not all(price_cols.values()):
    st.error('Please map all price columns in the sidebar (open, high, low, close).')
    st.stop()

# ---------- Processing ----------
with st.spinner('Aggregating daily OHLC...'):
    daily = to_daily_ohlc(df_raw, dt_col, price_cols)

st.subheader('Daily OHLC — sample')
st.dataframe(daily.tail(10))

# Bullish days
daily['bullish'] = daily['close'] > daily['open']

# HOD/LOD histograms
st.subheader('High/Low Distributions (bullish vs bearish)')
fig_high = px.histogram(daily, x='high', color=daily['bullish'].map({True:'bullish',False:'bearish'}), nbins=50, title='High values by day type')
fig_low = px.histogram(daily, x='low', color=daily['bullish'].map({True:'bullish',False:'bearish'}), nbins=50, title='Low values by day type')
col1, col2 = st.columns(2)
with col1: st.plotly_chart(fig_high, use_container_width=True)
with col2: st.plotly_chart(fig_low, use_container_width=True)

# Orderflow
st.subheader('Orderflow (if present)')
daily_of = compute_orderflow(df_raw, dt_col)
if daily_of is None:
    st.info('No recognized orderflow columns found. Provide buy_volume & sell_volume OR tick_direction & volume to compute orderflow.')
else:
    st.dataframe(daily_of.tail(10))
    st.line_chart(daily_of['delta'])
    st.line_chart(daily_of['imbalance'])

# Quantile Forecast
st.subheader('Quantile-based Forecast (next day)')
if len(quantile_list) == 0:
    st.warning('Select at least one quantile in the sidebar to compute the forecast.')
else:
    with st.spinner('Computing quantile forecast...'):
        qdf = quantile_forecast(daily, lookback=lookback, quantiles=sorted(quantile_list))
    st.dataframe(qdf)

# True Range Forecast
st.subheader('Next Day True Range Forecast (%)')
with st.spinner('Computing True Range forecast...'):
    tr_forecast = true_range_forecast(daily)
st.metric("Forecast TR%", f"{
