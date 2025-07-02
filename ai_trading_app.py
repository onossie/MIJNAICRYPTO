import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from python_bitvavo_api.bitvavo import Bitvavo

# Config en Bitvavo setup
API_KEY = st.secrets["BITVAVO_API_KEY"]
API_SECRET = st.secrets["BITVAVO_API_SECRET"]

bitvavo = Bitvavo({
    'APIKEY': API_KEY,
    'APISECRET': API_SECRET,
    'RESTURL': 'https://api.bitvavo.com/v2',
    'WSURL': 'wss://ws.bitvavo.com/v2/'
})

CANDLE_INTERVAL = "1h"
CANDLE_LIMIT = 100
ORDER_SIZE = 0.01  # grootte van de trade in coin eenheid
START_BALANCE = 1000.0  # startbedrag in EUR voor paper trading

# Technische indicatoren toevoegen
def add_technical_indicators(df):
    df = df.copy()
    df['SMA10'] = df['close'].rolling(10).mean()
    df['SMA30'] = df['close'].rolling(30).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    RS = gain / loss
    df['RSI14'] = 100 - (100 / (1 + RS))
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

# Data ophalen met caching
@st.cache_data(ttl=120)
def get_markets_eur():
    markets = bitvavo.markets()
    eur_markets = [m['market'] for m in markets if m['market'].endswith('EUR')]
    return eur_markets

@st.cache_data(ttl=120)
def get_candles(symbol):
    candles = bitvavo.candles(symbol, CANDLE_INTERVAL, {"limit": CANDLE_LIMIT})
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df = add_technical_indicators(df)
    return df

# Prepare features en target
def prepare_features_targets(df):
    df = df.copy()
    df['future_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    df['target'] = (df['future_close'] > df['close']).astype(int)  # 1 = prijs omhoog, 0 = omlaag of gelijk

    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'SMA10', 'SMA30', 'RSI14']
    X = df[feature_cols].values
    y = df['target'].values
    return X, y, feature_cols

# Simuleer trading met het model op testdata
def simulate_trading(model, scaler, X_test, y_test, prices_test):
    X_scaled = scaler.transform(X_test)
    preds = model.predict(X_scaled)

    balance = START_BALANCE
    position = None
    trades = []

    for i in range(len(preds)):
        price = prices_test[i]
        decision = preds[i]

        if decision == 1 and position is None:  # kopen
            position = price
            balance -= price * ORDER_SIZE
            trades.append(f"KOOP @ {price:.2f} EUR")
        elif decision == 0 and position is not None:  # verkopen
            profit = (price - position) * ORDER_SIZE
            balance += price * ORDER_SIZE
            trades.append(f"VERKOOP @ {price:.2f} EUR, winst {profit:.2f} EUR")
            position = None

    # Open positie aan einde sluiten tegen laatste prijs
    if position is not None:
        profit = (prices_test[-1] - position) * ORDER_SIZE
        balance += prices_test[-1] * ORDER_SIZE
        trades.append(f"VERKOOP (slot) @ {prices_test[-1]:.2f} EUR, winst {profit:.2f} EUR")

    return balance, trades

# UI en logica
st.title("🤖 Zelflerende AI Crypto Trading Bot - Paper Trading")

eur_markets = get_markets_eur()
st.sidebar.write(f"Aantal EUR-markets beschikbaar: {len(eur_markets)}")
max_coins = st.sidebar.slider("Max aantal coins om te trainen en testen", 1, min(30, len(eur_markets)), 5)

symbols = eur_markets[:max_coins]

for symbol in symbols:
    st.header(f"Coin: {symbol}")
    df = get_candles(symbol)
    X, y, feature_cols = prepare_features_targets(df)

    split_idx = int(len(X) * 0.9)  # 90% train, 10% test
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    prices_test = df['close'].values[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    balance_end, trades = simulate_trading(model, scaler, X_test, y_test, prices_test)
    winst_paper = balance_end - START_BALANCE

    st.write(f"Model accuracy test set: **{acc*100:.2f}%**")
    st.write(f"Simulatie paper trading resultaat: **{winst_paper:.2f} EUR winst** uit startkapitaal van {START_BALANCE} EUR")
    st.write("Trade geschiedenis (testperiode):")
    for trade in trades:
        st.write(f"- {trade}")

    st.divider()

st.write("🔔 Dit is alleen paper trading. Geen live trades worden uitgevoerd.")
st.write("🔔 Focus ligt nu op goede training en evaluatie van de AI modellen.")
