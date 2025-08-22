import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -------------------------------------------------------------
# Fetch OHLCV Data with robust column normalization
# -------------------------------------------------------------
def fetch_ohlcv(tickers: list[str], start: str, end: str) -> dict:
    """Return dict[ticker] -> DataFrame with columns [Open, High, Low, Close, Volume]"""
    data = {}
    for t in tickers:
        try:
            df = yf.download(
                t, start=start, end=end, interval="1d",
                auto_adjust=False, progress=False
            )
            if df.empty:
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join([c for c in col if c]) for col in df.columns]

            # Normalize column names
            col_map = {
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Adj Close": "AdjClose",
                "AdjClose": "AdjClose",
                "Close_Close": "Close",
                "Open_Open": "Open",
                "High_High": "High",
                "Low_Low": "Low",
                "Volume": "Volume",
                "Volume_Volume": "Volume"
            }
            df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})

            # Keep only required columns
            keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df = df[keep_cols]

            df.index.name = "Date"
            data[t] = df
        except Exception as e:
            st.warning(f"Download failed for {t}: {e}")
    return data

# -------------------------------------------------------------
# Feature Engineering
# -------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Close" not in out.columns:
        return out  # safety fallback

    out["ret_close"] = out["Close"].pct_change()
    out["sma_5"] = out["Close"].rolling(5).mean()
    out["sma_10"] = out["Close"].rolling(10).mean()
    out["sma_ratio"] = out["Close"] / out["sma_10"].squeeze()
    out["vol_ma5"] = out["Volume"].rolling(5).mean()
    out = out.dropna()
    return out

# -------------------------------------------------------------
# Build Dataset
# -------------------------------------------------------------
def build_dataset(data: dict) -> pd.DataFrame:
    dfs = []
    for ticker, df in data.items():
        fe = add_features(df)
        if fe.empty:
            continue
        fe["Ticker"] = ticker
        fe["Target"] = (fe["ret_close"].shift(-1) > 0).astype(int)
        dfs.append(fe)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs)

# -------------------------------------------------------------
# Train & Predict
# -------------------------------------------------------------
def train_and_predict(df: pd.DataFrame):
    features = ["ret_close", "sma_5", "sma_10", "sma_ratio", "vol_ma5"]
    df = df.dropna(subset=features + ["Target"])
    if df.empty:
        return None, None, None

    X = df[features]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, features, report

# -------------------------------------------------------------
# Streamlit App
# -------------------------------------------------------------
st.set_page_config(page_title="Nifty50 Intraday Predictor", layout="wide")
st.title("📈 Nifty50 Next-Day Intraday Buy Predictor")

st.sidebar.header("Settings")
start = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365))
end = st.sidebar.date_input("End Date", datetime.today())

# Sample NIFTY50 tickers (can be replaced with file upload)
def load_tickers():
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "HINDUNILVR.NS"
    ]

symbols = load_tickers()

if st.sidebar.button("Run Prediction"):
    st.write("⏳ Fetching data and building model...")
    data = fetch_ohlcv(symbols, str(start), str(end))
    all_df = build_dataset(data)

    if all_df.empty:
        st.error("No valid data available.")
    else:
        model, features, report = train_and_predict(all_df)
        if model is None:
            st.error("Model training failed. Not enough data.")
        else:
            st.subheader("Model Performance")
            st.json(report)

            latest = all_df.groupby("Ticker").tail(1)
            X_latest = latest[features]
            preds = model.predict(X_latest)
            latest["Prediction"] = preds

            buys = latest[latest["Prediction"] == 1][["Ticker", "Close"]]
            st.subheader("Recommended Buys for Next Day")
            st.dataframe(buys)

else:
    st.info("Press **Run Prediction** to start.")
