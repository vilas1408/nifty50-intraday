import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# =============================
# Data Fetcher with Debug
# =============================
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
                st.warning(f"⚠️ No data for {t}")
                continue

            # Flatten MultiIndex if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join([c for c in col if c]) for col in df.columns]

            col_map = {
                "Open": "Open", "High": "High", "Low": "Low",
                "Close": "Close", "Adj Close": "AdjClose",
                "AdjClose": "AdjClose", "Close_Close": "Close",
                "Open_Open": "Open", "High_High": "High",
                "Low_Low": "Low", "Volume": "Volume",
                "Volume_Volume": "Volume"
            }
            df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})
            keep_cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in keep_cols if c in df.columns]]

            if df.empty:
                st.warning(f"⚠️ Cleaned DataFrame empty for {t}, columns: {df.columns}")
                continue

            df.index.name = "Date"
            data[t] = df

            # Debug: show first 2 rows
            st.write(f"✅ Data for {t}", df.head(2))

        except Exception as e:
            st.warning(f"❌ Download failed for {t}: {e}")
    return data

# =============================
# Feature Engineering
# =============================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_close"] = out["Close"].pct_change()
    out["sma_10"] = out["Close"].rolling(10).mean()
    out["sma_20"] = out["Close"].rolling(20).mean()
    out["sma_ratio"] = out["Close"] / out["sma_10"]
    out["vol_ma"] = out["Volume"].rolling(5).mean()
    out.dropna(inplace=True)
    return out

# =============================
# Build Dataset
# =============================
def build_dataset(data: dict) -> pd.DataFrame:
    frames = []
    for t, df in data.items():
        fe = add_features(df)
        fe["ticker"] = t
        frames.append(fe)
    if frames:
        return pd.concat(frames)
    else:
        return pd.DataFrame()

# =============================
# Model Trainer
# =============================
def train_model(df: pd.DataFrame):
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    features = ["ret_close", "sma_ratio", "vol_ma"]
    X = df[features]
    y = df["target"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, features

# =============================
# Prediction Function
# =============================
def predict_next_day(model, features, df: pd.DataFrame):
    latest = df.iloc[-1:]
    X_pred = latest[features]
    prob = model.predict_proba(X_pred)[0, 1]
    return prob

# =============================
# Streamlit UI
# =============================
st.title("📈 Nifty50 Intraday Next-Day Predictor")
st.write("Predict which Nifty50 stocks may rise tomorrow.")

uploaded = st.file_uploader("Upload tickers.csv (with column 'symbol')", type=["csv"])

if uploaded:
    tickers = pd.read_csv(uploaded)["symbol"].tolist()
    st.write("Tickers:", tickers)

    start = (datetime.today() - timedelta(days=200)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")

    data = fetch_ohlcv(tickers, start, end)
    all_df = build_dataset(data)

    if all_df.empty:
        st.error("No valid data available after processing.")
    else:
        model, features = train_model(all_df)

        results = []
        for t, df in data.items():
            try:
                fe = add_features(df)
                prob = predict_next_day(model, features, fe)
                results.append({"Ticker": t, "Prob_Up": prob})
            except Exception as e:
                st.warning(f"Prediction failed for {t}: {e}")

        if results:
            res_df = pd.DataFrame(results).sort_values("Prob_Up", ascending=False)
            st.subheader("Predicted Top Stocks for Tomorrow")
            st.dataframe(res_df)
        else:
            st.error("No predictions could be made.")
