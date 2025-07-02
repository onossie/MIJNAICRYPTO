import streamlit as st
import pandas as pd
import time
from python_bitvavo_api.bitvavo import Bitvavo
from sklearn.linear_model import LogisticRegression
import numpy as np
import os

# Configuratie
st.set_page_config(page_title="AI Crypto Bot", layout="wide")

# API Config
API_KEY = st.secrets["BITVAVO_API_KEY"]
API_SECRET = st.secrets["BITVAVO_API_SECRET"]

bitvavo = Bitvavo({
    'APIKEY': API_KEY,
    'APISECRET': API_SECRET,
    'RESTURL': 'https://api.bitvavo.com/v2',
    'WSURL': 'wss://ws.bitvavo.com/v2/',
    'ACCESSWINDOW': 10000,
    'DEBUGGING': False
})

# Instellingen
TRADING_PAIR = "ETH-EUR"
INITIAL_BALANCE = 96.0
TRAINING_SIZE = 100  # Aantal datapunten om op te trainen

# AI Model
model = LogisticRegression()

@st.cache_data
def get_historical_prices(pair="ETH-EUR", interval="1h", limit=200):
    data = bitvavo.candles(pair, interval, {"limit": limit})
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df = df.astype(float)
    return df

def create_features(df):
    df["return"] = df["close"].pct_change()
    df["target"] = (df["return"].shift(-1) > 0).astype(int)
    df = df.dropna()
    return df[["return"]], df["target"]

# Simuleer paper trading
def paper_trade(predictions, df, balance):
    coin = 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and balance > 0:
            coin = balance / df["close"].iloc[i]
            balance = 0
        elif predictions[i] == 0 and coin > 0:
            balance = coin * df["close"].iloc[i]
            coin = 0
    return balance + coin * df["close"].iloc[-1]

# UI
st.title("🤖 Zelflerende AI Crypto Bot (ETH-EUR)")
st.write("Start als paper trading. Later kun je echte trading inschakelen.")

# Data ophalen
with st.spinner("⏳ Gegevens laden..."):
    df = get_historical_prices()
    X, y = create_features(df)

    # Trainen
    model.fit(X.tail(TRAINING_SIZE), y.tail(TRAINING_SIZE))

    # Voorspellen
    predictions = model.predict(X)

    # Simulatie
    end_balance = paper_trade(predictions[-TRAINING_SIZE:], df.tail(TRAINING_SIZE), INITIAL_BALANCE)

    # Laatste prijs
    current_price = df["close"].iloc[-1]

# Resultaten
col1, col2 = st.columns(2)
with col1:
    st.metric("💰 Beginsaldo", f"€{INITIAL_BALANCE:.2f}")
    st.metric("📈 Eindsaldo (paper)", f"€{end_balance:.2f}")
    st.metric("📊 ETH Prijs", f"€{current_price:.2f}")
with col2:
    st.line_chart(df[["close"]].rename(columns={"close": "ETH-EUR"}))

# Handmatige schakelaar
real_trade = st.toggle("🔁 Zet over naar echt traden", value=False)

if real_trade:
    st.warning("Echte trading is nu actief! (nog niet geïmplementeerd, alleen simulatie)")
    # hier komt later code voor echte trading
else:
    st.info("De AI leert nu via paper trading. Zet over naar echt traden zodra het model goed presteert.")
