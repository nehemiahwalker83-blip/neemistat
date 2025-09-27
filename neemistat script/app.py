import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================
# Helper function
# ==========================
def to_daily_ohlc(df, datetime_col, price_cols):
    # Ensure datetime column is in datetime format
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col])  # Drop rows where datetime failed to parse

    # Create a date column
    df["date"] = df[datetime_col].dt.date

    # Aggregate into OHLC per day
    daily = df.groupby("date").agg(
        open=(price_cols["open"], "first"),
        high=(price_cols["high"], "max"),
        low=(price_cols["low"], "min"),
        close=(price_cols["close"], "last"),
    ).reset_index()

    return daily

# ==========================
# Streamlit app
# ==========================
st.set_page_config(page_title="Neemistat", layout="wide")
st.title("Neemistat Trading Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

    # Show raw data preview
    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head())

    # Define your column mappings here
    dt_col = "datetime"   # change if your column has a different name
    price_cols = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close"
    }

    # Convert to daily OHLC
    daily = to_daily_ohlc(df_raw, dt_col, price_cols)

    st.subheader("Daily OHLC Data")
    st.dataframe(daily.head())

    # Plot OHLC lines
    fig_candle = px.line(
        daily,
        x="date",
        y=["open", "high", "low", "close"],
        title="Daily OHLC (lines)"
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # Example metrics
    latest = daily.iloc[-1]
    st.subheader("Latest Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Open", f"{latest['open']:.2f}")
    col2.metric("High", f"{latest['high']:.2f}")
    col3.metric("Low", f"{latest['low']:.2f}")
    col4.metric("Close", f"{latest['close']:.2f}")

else:
    st.info("Please upload a CSV file to get started.")

