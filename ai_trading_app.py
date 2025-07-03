import streamlit as st
import asyncio
import threading
import json
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from python_bitvavo_api.bitvavo import Bitvavo
import matplotlib.pyplot as plt
import joblib
import os

# === Config ===
API_KEY = st.secrets["BITVAVO_API_KEY"]
API_SECRET = st.secrets["BITVAVO_API_SECRET"]

bitvavo = Bitvavo({
    'APIKEY': API_KEY,
    'APISECRET': API_SECRET
})

WS_URL = "wss://ws.bitvavo.com/v2/"

MARKET = "ETH-EUR"  # start met één coin, later uitbreiden

MIN_CANDLES = 50
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

START_BALANCE = 96.0

# Trade parameters
TAKE_PROFIT_PCT = 0.02
STOP_LOSS_PCT = 0.015
TRAILING_STOP_GAP_PCT = 0.005

# === Globals (in sessiestate) ===
if 'candles' not in st.session_state:
    st.session_state.candles = []

if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

if 'balance' not in st.session_state:
    st.session_state.balance = START_BALANCE
if 'position' not in st.session_state:
    st.session_state.position = None  # dict: qty, entry_price, max_price

if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

# === Helpers ===

def save_model(model, scaler):
    joblib.dump(model, os.path.join(MODEL_DIR, f"{MARKET.replace('/', '-')}_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{MARKET.replace('/', '-')}_scaler.pkl"))

def load_model():
    model_path = os.path.join(MODEL_DIR, f"{MARKET.replace('/', '-')}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{MARKET.replace('/', '-')}_scaler.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

def train_model(df):
    df = df.copy()
    df['future_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    df['target'] = (df['future_close'] > df['close']).astype(int)

    X = df[["open", "high", "low", "close", "volume"]]
    y = df['target']

    if len(df) < MIN_CANDLES:
        return None, None, 0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=500)
    model.fit(X_scaled, y)

    accuracy = model.score(X_scaled, y)

    return model, scaler, accuracy

def predict_action(model, scaler, candle):
    X = pd.DataFrame([candle], columns=["open", "high", "low", "close", "volume"])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return pred  # 1 = buy, 0 = sell

# === Realtime WebSocket ===
import websockets

async def ws_candles():
    async with websockets.connect(WS_URL) as ws:
        subscribe_msg = {
            "action": "subscribe",
            "subscriptions": [
                {
                    "name": "candles",
                    "markets": [MARKET],
                    "interval": "1m"
                }
            ]
        }
        await ws.send(json.dumps(subscribe_msg))

        async for message in ws:
            data = json.loads(message)
            if isinstance(data, list):
                for entry in data:
                    if 'market' in entry and 'interval' in entry:
                        candle = entry
                        c = candle
                        # Candle info: start, open, high, low, close, volume, trades
                        # We pakken open, high, low, close, volume
                        open_ = float(c['open'])
                        high = float(c['high'])
                        low = float(c['low'])
                        close = float(c['close'])
                        volume = float(c['volume'])

                        candle_data = [open_, high, low, close, volume]

                        # Voeg toe aan sessie candles
                        st.session_state.candles.append(candle_data)

                        # Hou max 200 candles
                        if len(st.session_state.candles) > 200:
                            st.session_state.candles.pop(0)

# === Trading logic ===
def trade_logic():
    if len(st.session_state.candles) < MIN_CANDLES:
        return "Wachten op voldoende data..."

    df = pd.DataFrame(st.session_state.candles, columns=["open", "high", "low", "close", "volume"])

    # Train model indien nog niet gedaan of iedere 30 candles opnieuw
    if st.session_state.model is None or len(st.session_state.candles) % 30 == 0:
        model, scaler, acc = train_model(df)
        if model:
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.accuracy = acc
            save_model(model, scaler)

    if st.session_state.model is None:
        return "Model nog niet getraind."

    # Voorspel actie op laatste candle
    last_candle = st.session_state.candles[-1]
    action = predict_action(st.session_state.model, st.session_state.scaler, last_candle)

    price = last_candle[3]  # close prijs

    # Handel simulatie
    pos = st.session_state.position

    if action == 1 and pos is None and st.session_state.balance > price:
        # Koop
        qty = st.session_state.balance / price
        st.session_state.position = {
            "qty": qty,
            "entry_price": price,
            "max_price": price
        }
        st.session_state.balance -= qty * price
        st.session_state.trade_history.append({"type": "buy", "price": price, "qty": qty, "time": time.time()})
        return f"Gekocht {qty:.4f} {MARKET} voor €{price:.2f}"

    elif action == 0 and pos is not None:
        # Verkoop
        qty = pos['qty']
        st.session_state.balance += qty * price
        st.session_state.trade_history.append({"type": "sell", "price": price, "qty": qty, "time": time.time()})
        st.session_state.position = None
        return f"Verkocht {qty:.4f} {MARKET} voor €{price:.2f}"

    # Trailing stop check
    if pos is not None:
        if price > pos['max_price']:
            pos['max_price'] = price
        trailing_stop = pos['max_price'] * (1 - TRAILING_STOP_GAP_PCT)
        if price < trailing_stop:
            # Trailing stop triggered: verkoop
            qty = pos['qty']
            st.session_state.balance += qty * price
            st.session_state.trade_history.append({"type": "sell", "price": price, "qty": qty, "time": time.time()})
            st.session_state.position = None
            return f"Trailing stop triggered: Verkocht {qty:.4f} {MARKET} voor €{price:.2f}"

    return "Geen actie"

# === Run WebSocket in aparte thread zodat Streamlit niet blokkeert ===
def start_ws_loop():
    asyncio.run(ws_candles())

if 'ws_thread_started' not in st.session_state:
    ws_thread = threading.Thread(target=start_ws_loop, daemon=True)
    ws_thread.start()
    st.session_state.ws_thread_started = True

# === UI ===
st.title("🤖 Realtime AI Crypto Trading Bot (Paper Trading)")

status = trade_logic()
st.write(status)

st.subheader("💰 Balans en Positie")
st.write(f"Start balans: €{START_BALANCE:.2f}")
st.write(f"Huidige balans: €{st.session_state.balance:.2f}")

if st.session_state.position:
    pos = st.session_state.position
    st.write(f"Open positie: {pos['qty']:.4f} {MARKET} @ €{pos['entry_price']:.2f} (Max prijs: €{pos['max_price']:.2f})")
else:
    st.write("Geen open positie")

st.subheader("📊 Model accuracy")
if 'accuracy' in st.session_state:
   
