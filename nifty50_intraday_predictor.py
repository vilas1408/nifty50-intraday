# nifty50_intraday_predictor.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import plotly.graph_objects as go

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

def plot_chart(df, ticker):
    """Plot candlestick with SMA overlays"""
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["Datetime"],
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price"
    ))

    # Moving Averages
    if "SMA_5" in df.columns:
        fig.add_trace(go.Scatter(x=df["Datetime"], y=df["SMA_5"], line=dict(color="blue", width=1.5), name="SMA 5"))
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df["Datetime"], y=df["SMA_20"], line=dict(color="orange", width=1.5), name="SMA 20"))

    fig.update_layout(
        title=f"{ticker} - Candlestick Chart with SMA",
        xaxis_rangeslider_visible=False,
        height=500,
        template="plotly_white"
    )
    return fig

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📈 Nifty50 Intraday Predictor with Charts")
st.markdown("Predicts **next-day intraday BUY/SELL signals** from top Nifty50 stocks + charts.")

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
            "Signal": signal,
            "Data": df
        })

        progress.progress((i + 1) / len(tickers))
        time.sleep(0.2)  # avoid throttling

    if results:
        res_df = pd.DataFrame([{"Ticker": r["Ticker"], "Last Close": r["Last Close"], "Signal": r["Signal"]} for r in results])
        buy_signals = res_df[res_df["Signal"] == "BUY"].sort_values("Last Close", ascending=False)

        st.subheader("✅ Top 5 Stocks to BUY Next Day (Intraday)")
        st.table(buy_signals.head(5))

        st.subheader("📊 All Predictions")
        st.dataframe(res_df)

        # Chart Viewer
        st.subheader("📉 Candlestick Charts with Moving Averages")
        choice = st.selectbox("Select a stock to view chart", res_df["Ticker"].tolist())

        chart_data = next((r["Data"] for r in results if r["Ticker"] == choice), None)
        if chart_data is not None:
            fig = plot_chart(chart_data, choice)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No valid data fetched for any ticker.")
