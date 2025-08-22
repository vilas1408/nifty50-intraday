# nifty50_intraday_predictor.py
# Streamlit app to rank NIFTY 50 stocks for next-day intraday buys
# Disclaimer: Educational use only. Not investment advice.

import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import streamlit as st

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="NIFTY50 Next-Day Intraday Predictor", layout="wide")

st.title("NIFTY 50 — Next-Day Intraday Buy Candidates (Prototype)")
st.caption("Educational prototype. Uses daily data + technical features to classify whether tomorrow's intraday (Open→Close) return will be positive.")

# ---------------------------
# Helper: Load tickers
# ---------------------------
DEFAULT_TICKERS = [
    # Edit this list as needed. Add ".NS" suffix for NSE tickers on Yahoo Finance.
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","LT.NS","SBIN.NS","BHARTIARTL.NS",
    "ITC.NS","HINDUNILVR.NS","ASIANPAINT.NS","KOTAKBANK.NS","AXISBANK.NS","BAJFINANCE.NS","ULTRACEMCO.NS",
    "MARUTI.NS","SUNPHARMA.NS","HCLTECH.NS","TITAN.NS","WIPRO.NS","ONGC.NS","NTPC.NS","TATASTEEL.NS",
    "POWERGRID.NS","BAJAJFINSV.NS","ADANIENT.NS","GRASIM.NS","ADANIPORTS.NS","M&M.NS","JSWSTEEL.NS",
    "TATAMOTORS.NS","HDFCLIFE.NS","COALINDIA.NS","BRITANNIA.NS","TECHM.NS","DRREDDY.NS","HEROMOTOCO.NS",
    "CIPLA.NS","HINDALCO.NS","DIVISLAB.NS","NESTLEIND.NS","BPCL.NS","BAJAJ-AUTO.NS","EICHERMOT.NS",
    "TATACONSUM.NS","HAVELLS.NS","LTIM.NS","SHRIRAMFIN.NS","APOLLOHOSP.NS","TATAELXI.NS"
]

@st.cache_data(show_spinner=False)
def load_tickers(file: Path | None) -> list[str]:
    if file is not None:
        try:
            df = pd.read_csv(file)
            col = [c for c in df.columns if c.lower().strip() in {"ticker","tickers","symbol","symbols"}]
            if col:
                tickers = df[col[0]].dropna().astype(str).str.strip().tolist()
                return [t if t.endswith('.NS') else f"{t}.NS" for t in tickers]
        except Exception as e:
            st.warning(f"Failed to read uploaded tickers CSV: {e}")
    return DEFAULT_TICKERS

# ---------------------------
# Download data
# ---------------------------
@st.cache_data(show_spinner=True)
def fetch_ohlcv(tickers: list[str], start: str, end: str) -> dict:
    """Return dict[ticker] -> DataFrame with columns [Open, High, Low, Close, Volume]"""
    data = {}
    # yfinance multi-download can be messy; loop per ticker for robustness
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
            if df.empty:
                continue
            df = df.rename(columns=str.title)
            df.index.name = "Date"
            data[t] = df
        except Exception as e:
            st.warning(f"Download failed for {t}: {e}")
    return data

# ---------------------------
# Feature engineering
# ---------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Basic returns
    out["ret_close"] = out["Close"].pct_change()
    out["ret_open"] = out["Open"].pct_change()
    out["ret_oc"] = out["Close"] / out["Open"] - 1.0  # intraday
    out["gap"] = out["Open"] / out["Close"].shift(1) - 1.0

    # Volatility
    out["vol_10"] = out["ret_close"].rolling(10).std()
    out["vol_20"] = out["ret_close"].rolling(20).std()

    # Moving averages
    out["sma_5"] = out["Close"].rolling(5).mean()
    out["sma_10"] = out["Close"].rolling(10).mean()
    out["sma_20"] = out["Close"].rolling(20).mean()
    out["sma_ratio"] = out["Close"] / out["sma_10"]

    # TA indicators (ta lib is pure python)
    out["rsi_14"] = ta.momentum.RSIIndicator(out["Close"], window=14).rsi()
    macd = ta.trend.MACD(out["Close"], window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    atr = ta.volatility.AverageTrueRange(out["High"], out["Low"], out["Close"], window=14)
    out["atr_14"] = atr.average_true_range()

    adx = ta.trend.ADXIndicator(out["High"], out["Low"], out["Close"], window=14)
    out["adx_14"] = adx.adx()

    # Forward label: next-day intraday return (Open->Close) > 0
    out["ret_oc_next"] = out["ret_oc"].shift(-1)
    out["y"] = (out["ret_oc_next"] > 0).astype(int)

    return out

FEATURES = [
    "gap","ret_close","vol_10","vol_20","sma_ratio","rsi_14","macd","macd_signal","macd_hist",
    "atr_14","adx_14"
]

# ---------------------------
# Build pooled dataset
# ---------------------------

def build_dataset(data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for t, df in data_dict.items():
        fe = add_features(df)
        fe = fe.reset_index()
        fe.insert(1, "ticker", t)
        frames.append(fe)
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=FEATURES + ["y"])  # drop rows with NaNs in features/label
    return all_df

# ---------------------------
# Train model (pooled across tickers)
# ---------------------------

def train_model(all_df: pd.DataFrame) -> tuple[Pipeline, dict]:
    all_df = all_df.sort_values("Date")

    X = all_df[["ticker"] + FEATURES].copy()
    y = all_df["y"].astype(int).values

    # One-hot encode ticker, passthrough numerical features
    pre = ColumnTransformer(
        transformers=[
            ("tick", OneHotEncoder(handle_unknown="ignore"), ["ticker"]),
            ("num", "passthrough", FEATURES),
        ]
    )

    clf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=5, n_jobs=-1, random_state=42)

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    # TimeSeries CV metrics
    tscv = TimeSeriesSplit(n_splits=5)
    aucs, accs = [], []
    for tr_idx, te_idx in tscv.split(X):
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        proba = pipe.predict_proba(X.iloc[te_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        try:
            aucs.append(roc_auc_score(y[te_idx], proba))
        except Exception:
            pass
        accs.append(accuracy_score(y[te_idx], pred))

    metrics = {
        "cv_auc_mean": float(np.mean(aucs)) if aucs else np.nan,
        "cv_acc_mean": float(np.mean(accs)) if accs else np.nan,
        "splits": len(accs),
    }

    # Fit on all data
    pipe.fit(X, y)
    return pipe, metrics

# ---------------------------
# Generate tomorrow predictions
# ---------------------------

def latest_signals(model: Pipeline, data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    today = None
    for t, df in data_dict.items():
        fe = add_features(df)
        if fe.empty:
            continue
        last_row = fe.iloc[-1]
        # Prepare a single-row X for prediction using today's features
        Xrow = pd.DataFrame({"ticker": [t], **{f: [last_row[f]] for f in FEATURES}})
        proba = model.predict_proba(Xrow)[:, 1][0]

        # Use ATR for position sizing & stop suggestions
        atr = float(last_row.get("atr_14", np.nan))
        last_close = float(last_row["Close"]) if "Close" in last_row else float(df["Close"].iloc[-1])
        last_open = float(last_row["Open"]) if "Open" in last_row else float(df["Open"].iloc[-1])

        rows.append({
            "ticker": t,
            "date": fe.index[-1] if isinstance(fe.index, pd.DatetimeIndex) else df.index[-1],
            "proba_up_next_day": float(proba),
            "close": last_close,
            "open": last_open,
            "atr14": float(atr),
            "suggested_stop": round(last_close - 1.0 * atr if not np.isnan(atr) else np.nan, 2),
            "suggested_target": round(last_close + 1.5 * atr if not np.isnan(atr) else np.nan, 2),
        })
        today = rows[-1]["date"]

    sigs = pd.DataFrame(rows)
    if not sigs.empty:
        sigs = sigs.sort_values("proba_up_next_day", ascending=False)
    return sigs, today

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Universe & Data")
    uploaded = st.file_uploader("Upload tickers CSV (column: ticker)", type=["csv"])
    tickers = load_tickers(uploaded)

    years = st.slider("Years of history", 1, 5, 3)
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=365*years + 10)

    st.markdown("---")
    st.header("Signal Filter")
    min_proba = st.slider("Minimum probability (up)", 0.5, 0.9, 0.65, 0.01)
    top_n = st.slider("Top N picks", 1, 20, 10)

    st.markdown("---")
    st.header("Risk Settings (suggestions)")
    risk_capital = st.number_input("Risk capital (₹)", min_value=1000, value=50000, step=1000)
    risk_per_trade = st.number_input("Risk per trade (%)", min_value=0.1, value=1.0, step=0.1)
    atr_mult = st.slider("Stop ATR multiple", 0.5, 3.0, 1.0, 0.1)

# ---------------------------
# Main flow
# ---------------------------
with st.spinner("Downloading data & training model..."):
    data = fetch_ohlcv(tickers, start=start_date.strftime('%Y-%m-%d'), end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'))
    if not data:
        st.error("No data fetched. Check tickers or internet connectivity.")
        st.stop()

    all_df = build_dataset(data)
    if all_df.empty:
        st.error("No usable rows after feature engineering.")
        st.stop()

    model, metrics = train_model(all_df)

st.subheader("Cross-validated performance (pooled)")
col1, col2, col3 = st.columns(3)
col1.metric("Splits", metrics.get("splits", 0))
col2.metric("CV AUC", f"{metrics.get('cv_auc_mean', float('nan')):.3f}")
col3.metric("CV Accuracy", f"{metrics.get('cv_acc_mean', float('nan')):.3f}")

sigs, last_date = latest_signals(model, data)

if sigs.empty:
    st.warning("Could not compute signals.")
    st.stop()

st.subheader(f"Signals for {pd.to_datetime(last_date).date() if last_date is not None else 'latest date'}")

# Position sizing based on ATR
if "atr14" in sigs.columns:
    risk_rupees = risk_capital * (risk_per_trade / 100.0)
    # stop distance uses atr_mult * ATR
    sigs["stop_distance"] = sigs["atr14"] * atr_mult
    sigs["qty"] = (risk_rupees / sigs["stop_distance"]).clip(lower=0).round()

# Filter & pick top N
view = sigs.query("proba_up_next_day >= @min_proba").copy()
view = view.head(top_n)

st.dataframe(view[["ticker","proba_up_next_day","close","atr14","suggested_stop","suggested_target","qty"]], use_container_width=True)

st.markdown("""
**How to interpret**
- *proba_up_next_day*: Model's probability that tomorrow's intraday (Open→Close) return is positive.
- *suggested_stop/target*: Simple ATR-based markers from today's Close (not a recommendation).
- *qty*: Risk-based position size using your settings (rounded shares).

**Important**: This is a simplified classification model on end-of-day data. Real intraday trading requires robust execution, slippage, transaction costs, and risk management.
""")

with st.expander("Show raw engineered dataset (sample)"):
    st.dataframe(all_df.tail(500), use_container_width=True)

st.markdown("---")
st.caption("© 2025 Prototype. Educational use only. Not investment advice.")
