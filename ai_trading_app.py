import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from python_bitvavo_api.bitvavo import Bitvavo
import time

# =======================
# CONFIGURATIE & API KEY
# =======================
API_KEY = st.secrets["BITVAVO_API_KEY"]
API_SECRET = st.secrets["BITVAVO_API_SECRET"]

bitvavo = Bitvavo({
    'APIKEY': API_KEY,
    'APISECRET': API_SECRET,
    'RESTURL': 'https://api.bitvavo.com/v2',
    'WSURL': 'wss://ws.bitvavo.com/v2/'
})

START_BALANCE = 96.0
MIN_CANDLES = 50  # minimaal aantal candles per coin om te trainen
MAX_COINS = 385   # test maximaal 385 coins
paper_trading = True

# =======================
# FUNCTIES
# =======================
@st.cache_data(ttl=3600)
def get_all_eur_markets():
    markets = bitvavo.markets()
    return [m['market'] for m in markets if m['quote'] == 'EUR'][:MAX_COINS]

def get_historical_prices(symbol, interval='1h', limit=100):
    candles = bitvavo.candles(symbol, interval, {"limit": limit})
    if len(candles) < MIN_CANDLES:
        return None
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def train_ai_model(df):
    df['future_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    df['target'] = (df['future_close'] > df['close']).astype(int)

    X = df[["open", "high", "low", "close", "volume"]]
    y = df['target']

    if len(df) < 20:
        return None, None

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler

def make_decision(model, scaler, latest_data):
    X = latest_data[["open", "high", "low", "close", "volume"]]
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    return prediction[-1]

# =======================
# STREAMLIT UI
# =======================
st.title("🤖 AI Crypto Trading Bot (Paper Trading)")
st.markdown(f"**Startsaldo:** €{START_BALANCE:.2f} | 🧪 Paper Trading Mode")

coin_list = get_all_eur_markets()
balance_per_coin = START_BALANCE / len(coin_list)
total_profit = 0
results = []

progress_bar = st.progress(0)
status = st.empty()

for idx, symbol in enumerate(coin_list):
    status.text(f"Bezig met: {symbol} ({idx+1}/{len(coin_list)})")
    df = get_historical_prices(symbol)

    if df is None:
        continue

    model, scaler = train_ai_model(df)

    if model is None:
        continue

    latest = df.tail(1)
    decision = make_decision(model, scaler, latest)

    entry_price = latest['close'].values[0]
    profit = 0

    # Simuleer trade
    if decision == 1:  # koop
        exit_price = df['close'].values[-1]  # laatste prijs als verkoop
        profit = exit_price - entry_price
        total_profit += profit
        results.append({
            'coin': symbol,
            'buy': entry_price,
            'sell': exit_price,
            'profit': profit
        })

    progress_bar.progress((idx + 1) / len(coin_list))

# =======================
# RESULTATEN
# =======================
st.subheader("📊 Resultaten")
if results:
    result_df = pd.DataFrame(results)
    result_df['profit(€)'] = result_df['profit']
    st.dataframe(result_df[["coin", "buy", "sell", "profit(€)"]].sort_values(by="profit(€)", ascending=False))
