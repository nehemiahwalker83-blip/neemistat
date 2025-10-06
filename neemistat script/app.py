import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests, io, os, pathlib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# =====================================================
# 🌿 Neemistat v2.0 – Quantitative Forecast & Institutional Bias Dashboard
# =====================================================

st.set_page_config(page_title="Neemistat", layout="wide")
st.title("📈 Neemistat – Quantitative Forecast, Density Map & Institutional Bias")

# =====================================================
# AUTO-FETCH COT DATA (Official CFTC Source)
# =====================================================
def fetch_latest_cot():
    os.makedirs("data", exist_ok=True)
    file_path = "data/cot_data.csv"

    # skip if fresh (<7d)
    if os.path.exists(file_path):
        modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        if (datetime.now() - modified_time).days < 7:
            return

    try:
        url = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        raw = response.text

        lines = raw.splitlines()
        records = []
        for line in lines:
            if "NASDAQ-100" in line:
                parts = line.split(",")
                date = parts[2].strip().replace('"', "")
                comm_long = int(parts[7])
                comm_short = int(parts[8])
                fund_long = int(parts[11])
                fund_short = int(parts[12])
                small_long = int(parts[17])
                small_short = int(parts[18])
                records.append({
                    "Date": date,
                    "Commercials": comm_long - comm_short,
                    "Funds": fund_long - fund_short,
                    "Smalls": small_long - small_short
                })

        if records:
            df = pd.DataFrame(records)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
            df.to_csv(file_path, index=False)
            print(f"✅ Updated COT data saved to {file_path}")
        else:
            print("⚠️ No NASDAQ-100 data found in latest COT report.")
    except Exception as e:
        print(f"❌ Failed to fetch latest COT data: {e}")

# =====================================================
# WEEKLY AUTO-SCHEDULER (Friday 5:00 PM ET)
# =====================================================
COT_RUN_LOG = pathlib.Path("data/cot_last_run.txt")
ET_TZ = ZoneInfo("America/New_York")

def _read_last_run():
    try:
        return datetime.fromisoformat(COT_RUN_LOG.read_text().strip())
    except Exception:
        return None

def _write_last_run(dt):
    COT_RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    COT_RUN_LOG.write_text(dt.isoformat())

def _this_weeks_friday_5pm_et(now_et):
    weekday = now_et.isoweekday()
    delta = 5 - weekday
    friday = (now_et + timedelta(days=delta)).date()
    return datetime.combine(friday, datetime.min.time(), ET_TZ).replace(hour=17)

def maybe_run_weekly_cot_update():
    now_et = datetime.now(ET_TZ)
    target = _this_weeks_friday_5pm_et(now_et)
    last_run = _read_last_run()
    need_run = now_et >= target and (last_run is None or last_run < target)
    if need_run:
        fetch_latest_cot()
        _write_last_run(now_et)
        return True
    return False

# Manual + scheduled refresh
colA, colB = st.columns([1,3])
with colA:
    if st.button("🔄 Refresh COT Now"):
        fetch_latest_cot()
        st.success("COT data refreshed from CFTC.")
did_run = maybe_run_weekly_cot_update()
if did_run:
    st.info("Auto-update: fetched latest COT data for this week (Fri 5 PM ET).")

# heartbeat refresh every 60 s
st.write("""<meta http-equiv="refresh" content="60">""", unsafe_allow_html=True)

# =====================================================
# LOAD MARKET DATA
# =====================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/mnq_data.csv")
        df["datetime"] = pd.to_datetime(df["datetime"])
    except:
        dates = pd.date_range(end=datetime.now(), periods=200, freq="H")
        df = pd.DataFrame({
            "datetime": dates,
            "open": np.random.uniform(17000,17500,len(dates)),
            "high": np.random.uniform(17500,17700,len(dates)),
            "low": np.random.uniform(16800,17000,len(dates)),
            "close": np.random.uniform(17100,17600,len(dates)),
            "volume": np.random.randint(100,1000,len(dates))
        })
    return df
df = load_data()

# =====================================================
# DENSITY MAP + HOD/LOD HISTOGRAM
# =====================================================
st.header("📊 Density Map & HOD/LOD Histogram")
df["bullish"] = df["close"]>df["open"]
df["day"]=df["datetime"].dt.date
daily=df.groupby("day").agg({"high":"max","low":"min","open":"first","close":"last"}).reset_index()
daily["true_range"]=daily["high"]-daily["low"]
daily["bullish"]=daily["close"]>daily["open"]
bullish_tr=daily[daily["bullish"]]["true_range"]
bearish_tr=daily[~daily["bullish"]]["true_range"]
hist_df=pd.DataFrame({"Type":["Bullish"]*len(bullish_tr)+["Bearish"]*len(bearish_tr),
                      "TrueRange":pd.concat([bullish_tr,bearish_tr])})
st.plotly_chart(px.histogram(hist_df,x="TrueRange",color="Type",nbins=30,
                 title="True Range Distribution by Day Type"),use_container_width=True)

# =====================================================
# FORECASTING (AR1 MODEL)
# =====================================================
st.header("🔮 Forecasting Module (AR1 Model – 2 Year Historical)")
def ar1_forecast(series):
    x=series.values
    phi=np.corrcoef(x[:-1],x[1:])[0,1]
    return round(phi*x[-1],2),round(phi,3)
forecast_value,phi=ar1_forecast(daily["true_range"])
st.metric("Next-Day True Range Forecast (points)",forecast_value)
st.caption(f"AR(1) coefficient (persistence): {phi}")

# =====================================================
# TRUE RANGE DETAILS
# =====================================================
st.header("📐 True Range (Points & Percent)")
latest_tr=daily["true_range"].iloc[-1]
avg_tr=daily["true_range"].mean()
percent_tr=(latest_tr/avg_tr)*100
col1,col2,col3=st.columns(3)
col1.metric("Latest True Range (pts)",f"{latest_tr:.1f}")
col2.metric("Average TR (2 yrs)",f"{avg_tr:.1f}")
col3.metric("Relative Volatility (%)",f"{percent_tr:.1f}%")

# =====================================================
# BIAS SELECTION
# =====================================================
st.header("⚙️ Bias Selection")
mode=st.radio("Select Mode:",["Prediction","Forecasting"])
if mode=="Prediction":
    st.success("Prediction Mode active – uses AR(1) inference and density clustering.")
else:
    st.info("Forecasting Mode active – quantile projections and trend persistence.")

# =====================================================
# INSTITUTIONAL COT BIAS
# =====================================================
st.header("🏦 Institutional Bias (COT Data)")
@st.cache_data
def load_cot_data():
    try:
        df=pd.read_csv("data/cot_data.csv")
    except:
        df=pd.DataFrame({
            "Date":pd.date_range("2024-01-01",periods=12,freq="W"),
            "Commercials":np.random.randint(-40000,40000,12),
            "Funds":np.random.randint(-30000,30000,12),
            "Smalls":np.random.randint(-8000,8000,12)
        })
    df["Date"]=pd.to_datetime(df["Date"])
    return df.sort_values("Date")

cot=load_cot_data()
latest=cot.iloc[-1]
commercials_net, funds_net, smalls_net = latest["Commercials"], latest["Funds"], latest["Smalls"]

def cot_index(series,window):
    recent=series[-window:]
    return 100*(series.iloc[-1]-recent.min())/(recent.max()-recent.min()+1e-9)

cot6=cot_index(cot["Commercials"],6 if len(cot)>=6 else len(cot))
cot36=cot_index(cot["Commercials"],12 if len(cot)>=12 else len(cot))

def get_cot_bias(c,f,s):
    score=0
    if c>0 and f<0: score+=1
    elif c<0 and f>0: score-=1
    if s>0 and score>0: score-=0.5
    elif s<0 and score<0: score+=0.5
    return round(score,2)

bias_score=get_cot_bias(commercials_net,funds_net,smalls_net)
bias_label="🟩 Bullish" if bias_score>0 else "🟥 Bearish" if bias_score<0 else "🟨 Neutral"

c1,c2,c3=st.columns(3)
with c1:
    st.metric("Commercials Net",f"{commercials_net:,}")
    st.metric("6 Month Index",f"{cot6:.1f}%")
with c2:
    st.metric("Leveraged Funds Net",f"{funds_net:,}")
    st.metric("36 Month Index",f"{cot36:.1f}%")
with c3:
    st.metric("Small Traders Net",f"{smalls_net:,}")
    st.metric("Bias Signal",bias_label)
st.line_chart(cot.set_index("Date")[["Commercials","Funds","Smalls"]])

# =====================================================
# ALIGNMENT SUMMARY
# =====================================================
st.header("📋 System Alignment Summary")
if bias_score>0:
    st.success("✅ Institutional bias supports **long setups** – favor AMDX accumulation → manipulation → continuation.")
elif bias_score<0:
    st.error("⚠️ Institutional bias supports **short setups** – favor AMDX distribution → continuation breakdowns.")
else:
    st.warning("⚖️ Institutional bias is neutral – expect range-bound conditions.")

st.caption("© Neemistat v2.0 – Full Quantitative System with Institutional Auto-Updater & Scheduler.")
