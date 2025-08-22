import streamlit as st
import pandas as pd
import ta
from datetime import datetime, timedelta
from upstox_client import ApiClient, Configuration, MarketQuoteApi, HistoryApi

# -----------------------------
# Load Upstox credentials
# -----------------------------
api_key = st.secrets["upstox"]["api_key"]
api_secret = st.secrets["upstox"]["api_secret"]
redirect_uri = st.secrets["upstox"]["redirect_uri"]
access_token = st.secrets["upstox"]["access_token"]

# Configure Upstox API
config = Configuration()
config.access_token = access_token
api_client = ApiClient(config)

history_api = HistoryApi(api_client)
quote_api = MarketQuoteApi(api_client)

# -----------------------------
# Nifty50 Stock List
# -----------------------------
NIFTY50 = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "KOTAKBANK",
    "SBIN", "AXISBANK", "LT", "BHARTIARTL", "ITC", "HINDUNILVR",
    "ASIANPAINT", "HCLTECH", "SUNPHARMA", "TITAN", "MARUTI", "WIPRO",
    "BAJFINANCE", "ADANIPORTS", "ONGC", "POWERGRID", "NTPC", "NESTLEIND",
    "TATASTEEL", "JSWSTEEL", "ULTRACEMCO", "GRASIM", "HEROMOTOCO",
    "EICHERMOT", "DRREDDY", "CIPLA", "DIVISLAB", "BAJAJFINSV",
    "BRITANNIA", "M&M", "HINDALCO", "TECHM", "COALINDIA", "BPCL",
    "IOC", "SHREECEM", "ADANIENT", "LTIM", "UPL", "INDUSINDBK",
    "HDFCLIFE", "APOLLOHOSP"
]

# -----------------------------
# Fetch Historical Data
# -----------------------------
def fetch_stock_data(symbol, interval="5minute", days=5):
    """
    Fetch OHLCV data for given symbol using Upstox History API.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        response = history_api.get_historical_candle_data1(
            instrument_key=f"NSE_EQ|{symbol}",
            interval=interval,
            to_date=end_date.strftime("%Y-%m-%d"),
            from_date=start_date.strftime("%Y-%m-%d")
        )

        candles = response.data.candles
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"])
        return df.sort_values("time").reset_index(drop=True)

    except Exception as e:
        st.warning(f"⚠️ Error fetching {symbol}: {e}")
        return pd.DataFrame()

# -----------------------------
# Add Technical Indicators
# -----------------------------
def add_indicators(df):
    if df.empty:
        return df

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["sma20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    df["sma50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()
    return df

# -----------------------------
# Generate Buy/Sell Signals
# -----------------------------
def generate_signals(df):
    if df.empty:
        return None

    latest = df.iloc[-1]
    if latest["rsi"] < 30 and latest["macd"] > latest["macd_signal"] and latest["sma20"] > latest["sma50"]:
        return "BUY"
    elif latest["rsi"] > 70 and latest["macd"] < latest["macd_signal"] and latest["sma20"] < latest["sma50"]:
        return "SELL"
    else:
        return "HOLD"

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Nifty50 Intraday Predictor", layout="wide")

st.title("📈 Nifty50 Intraday Stock Predictor (Upstox API)")

st.sidebar.header("Settings")
interval = st.sidebar.selectbox("Select Interval", ["1minute", "5minute", "15minute", "30minute", "day"], index=1)
days = st.sidebar.slider("Number of past days", 1, 30, 5)

if st.sidebar.button("Run Analysis"):
    results = []

    with st.spinner("Fetching data from Upstox..."):
        for symbol in NIFTY50:
            df = fetch_stock_data(symbol, interval=interval, days=days)
            if df.empty:
                continue
            df = add_indicators(df)
            signal = generate_signals(df)
            results.append({"Symbol": symbol, "Signal": signal, "Last Price": df["close"].iloc[-1]})

    if results:
        df_results = pd.DataFrame(results)
        st.subheader("📊 Stock Predictions")
        st.dataframe(df_results)

        st.subheader("✅ Suggested BUY Stocks")
        st.table(df_results[df_results["Signal"] == "BUY"])

    else:
        st.error("No data available. Please check API credentials or try again later.")
