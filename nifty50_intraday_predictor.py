# nifty50_intraday_predictor.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time

st.set_page_config(page_title="Nifty50 Intraday Predictor", layout="wide")

# ------------------------------
# Utility Functions
# ------------------------------
def load_tickers(filename="tickers.csv"):
    """Load tickers from CSV"""
    try:
        tickers_df = pd.read_csv(filename)
        return tickers_df["Symbol"].tolist()
    except Exception as e:
        st.error(f"Error loading tickers: {e}")
        return []

def safe_download(ticker, period="5d", interval="5m"):
    """Download stock data safely from Yahoo Finance"""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            st.warning(f"⚠️ No data for {ticker}")
            return None
        df = df.reset_index()
        df["Ticker"] = ticker
        return df
    except Exception as e:
        st.error(f"❌ Error fetching {ticker}: {e}")
        return None

def add_features(df):
    """Feature engineering for signals"""
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    out["SMA_5"] = out["Close"].rolling(5).mean()
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["Signal"] = np.where(out["SMA_5"] > out["SMA_20"], 1, -1)
    return out

def predict_next_day(df):
    """Simple rule-based prediction"""
    try:
        latest = df.iloc[-1]
        return "BUY" if latest["Signal"] == 1 else "SELL"
    except Exception:
        return "HOLD"

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📈 Nifty50 Intraday Predictor")
st.markdown("Predicts **next-day intraday BUY/SELL signals** from top Nifty50 stocks.")

tickers = load_tickers("tickers.csv")

if not tickers:
    st.error("No tickers found in tickers.csv")
else:
    st.info(f"Loaded {len(tickers)} tickers from CSV")

    results = []
    progress = st.progress(0)

    for i, ticker in enumerate(tickers):
        df = safe_download(ticker, period="5d", interval="5m")
        if df is None:
            continue

        df = add_features(df)
        signal = predict_next_day(df)

        results.append({
            "Ticker": ticker,
            "Last Close": round(df["Close"].iloc[-1], 2),
            "Signal": signal
        })

        progress.progress((i + 1) / len(tickers))
        time.sleep(0.5)  # avoid throttling

    if results:
        res_df = pd.DataFrame(results)
        buy_signals = res_df[res_df["Signal"] == "BUY"].sort_values("Last Close", ascending=False)

        st.subheader("✅ Top 5 Stocks to BUY Next Day (Intraday)")
        st.table(buy_signals.head(5))

        st.subheader("📊 All Predictions")
        st.dataframe(res_df)
    else:
        st.warning("No valid data fetched for any ticker.")
