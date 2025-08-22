# nifty50_intraday_predictor.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# Try importing Plotly; fallback to Matplotlib if not available
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    import matplotlib.pyplot as plt
    HAS_PLOTLY = False

# -------------------------------
# Load NIFTY50 tickers
# -------------------------------
NIFTY50_TICKERS = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS",
    "BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS","BPCL.NS","BHARTIARTL.NS",
    "BRITANNIA.NS","CIPLA.NS","COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS",
    "EICHERMOT.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS",
    "HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","ITC.NS",
    "INDUSINDBK.NS","INFY.NS","JSWSTEEL.NS","KOTAKBANK.NS","LTIM.NS",
    "LT.NS","M&M.NS","MARUTI.NS","NTPC.NS","NESTLEIND.NS","ONGC.NS",
    "POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS","SUNPHARMA.NS",
    "TCS.NS","TATACONSUM.NS","TATAMOTORS.NS","TATASTEEL.NS","TECHM.NS",
    "TITAN.NS","ULTRACEMCO.NS","WIPRO.NS"
]

# -------------------------------
# Data Fetch Function
# -------------------------------
@st.cache_data
def get_data(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, group_by="ticker")
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df = df.rename(columns=str.title)  # Normalize column names
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(df.columns):
            return pd.DataFrame()
        return df
    except Exception as e:
        st.warning(f"⚠️ Could not fetch {ticker}: {e}")
        return pd.DataFrame()

# -------------------------------
# Feature Engineering
# -------------------------------
def add_features(df):
    if df.empty:
        return df
    out = df.copy()
    out["ret_close"] = out["Close"].pct_change()
    out["sma_10"] = out["Close"].rolling(10).mean()
    out["sma_ratio"] = out["Close"] / out["sma_10"]
    out["volatility"] = out["ret_close"].rolling(10).std()
    out = out.dropna()
    return out

# -------------------------------
# Prediction Logic (Simple Rule)
# -------------------------------
def predict_signal(df):
    if df.empty:
        return "NO DATA"
    last = df.iloc[-1]
    if last["Close"] > last["sma_10"] and last["ret_close"] > 0:
        return "BUY"
    elif last["Close"] < last["sma_10"] and last["ret_close"] < 0:
        return "SELL"
    else:
        return "HOLD"

# -------------------------------
# Streamlit App
# -------------------------------
st.title("📈 NIFTY50 Intraday Predictor")
st.markdown("Predict intraday **Buy / Sell / Hold** signals for NIFTY50 stocks")

ticker = st.selectbox("Select a NIFTY50 stock", NIFTY50_TICKERS)

# Fetch data
df = get_data(ticker, period="6mo", interval="1d")

if df.empty:
    st.error(f"No data available for {ticker}")
else:
    fe_df = add_features(df)
    signal = predict_signal(fe_df)
    st.subheader(f"Prediction for next day: **{signal}**")

    # Chart
    st.subheader("Price Chart")
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["Date"],
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Candlestick"
        ))
        fig.update_layout(title=f"{ticker} Price Chart", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["Date"], df["Close"], label="Close Price")
        ax.plot(fe_df["Date"], fe_df["sma_10"], label="SMA 10", linestyle="--")
        ax.set_title(f"{ticker} Price Chart")
        ax.legend()
        st.pyplot(fig)

# -------------------------------
# Show Table
# -------------------------------
if not df.empty:
    st.subheader("Recent Data")
    st.dataframe(df.tail(10))
