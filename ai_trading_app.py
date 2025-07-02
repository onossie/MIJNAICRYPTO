import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from python_bitvavo_api.bitvavo import Bitvavo

# ====== CONFIG =======
API_KEY = st.secrets["BITVAVO_API_KEY"]
API_SECRET = st.secrets["BITVAVO_API_SECRET"]

bitvavo = Bitvavo({
    'APIKEY': API_KEY,
    'APISECRET': API_SECRET,
    'RESTURL': 'https://api.bitvavo.com/v2',
    'WSURL': 'wss://ws.bitvavo.com/v2/'
})

START_BALANCE = 96.0
paper_trading = True

MIN_CANDLES = 50

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ====== Helper functies =======
def get_all_eur_markets():
    markets = bitvavo.markets()
    coins = [m['market'] for m in markets if m['quote'] == 'EUR']
    return coins  # Geen limiet, alle EUR coins

def get_historical_prices(symbol, interval='1h', limit=100):
    try:
        candles = bitvavo.candles(symbol, interval, {"limit": limit})
        if len(candles) < MIN_CANDLES:
            return None
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception:
        return None

def train_model(df):
    df = df.copy()
    df['future_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    df['target'] = (df['future_close'] > df['close']).astype(int)

    X = df[["open", "high", "low", "close", "volume"]]
    y = df['target']

    if len(df) < 20:
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train_scaled, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

    return model, scaler, accuracy

def model_path(coin):
    safe_coin = coin.replace("/", "-")
    return os.path.join(MODEL_DIR, f"{safe_coin}_model.pkl"), os.path.join(MODEL_DIR, f"{safe_coin}_scaler.pkl")

def load_model_and_scaler(coin):
    model_file, scaler_file = model_path(coin)
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        return model, scaler
    return None, None

def save_model_and_scaler(coin, model, scaler):
    model_file, scaler_file = model_path(coin)
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

def train_or_load_model(df, coin):
    model, scaler = load_model_and_scaler(coin)
    if model is not None and scaler is not None:
        return model, scaler

    model, scaler, accuracy = train_model(df)
    if model and scaler:
        save_model_and_scaler(coin, model, scaler)
    return model, scaler

def predict_action(model, scaler, latest_row):
    X = latest_row[["open", "high", "low", "close", "volume"]]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[-1]
    return pred  # 1 = buy, 0 = sell

# ====== Streamlit UI =======
st.title("🤖 AI Crypto Trading Bot — Self-learning met budget €96")
st.markdown("Paper trading modus actief — het startbudget is €96.")

if 'balance' not in st.session_state:
    st.session_state.balance = START_BALANCE

if 'open_positions' not in st.session_state:
    st.session_state.open_positions = []

if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

st.sidebar.header("Handmatige controls")
if st.sidebar.button("Reset paper trading"):
    st.session_state.balance = START_BALANCE
    st.session_state.open_positions = []
    st.session_state.trade_history = []

coin_list = get_all_eur_markets()

st.subheader("📈 AI Trading Actie")

best_coin = None
best_signal = None
best_price = None
best_accuracy = 0

progress_text = st.empty()
progress_bar = st.progress(0)

for i, coin in enumerate(coin_list):
    progress_text.text(f"Analyseer {coin} ({i+1}/{len(coin_list)})")
    df = get_historical_prices(coin)
    if df is None:
        continue

    model, scaler = train_or_load_model(df, coin)
    if model is None or scaler is None:
        continue

    latest = df.tail(1)
    decision = predict_action(model, scaler, latest)

    current_price = latest['close'].values[0]

    if decision == 1 and st.session_state.balance >= current_price and best_accuracy < 1.0:
        best_coin = coin
        best_signal = "buy"
        best_price = current_price
        best_accuracy = 1.0

    elif decision == 0:
        held_positions = [pos for pos in st.session_state.open_positions if pos['coin'] == coin]
        if held_positions and best_accuracy < 1.0:
            best_coin = coin
            best_signal = "sell"
            best_price = current_price
            best_accuracy = 1.0

    progress_bar.progress((i + 1) / len(coin_list))
    time.sleep(0.05)

progress_text.text("Klaar met analyseren.")

st.subheader("📊 Trade Beslissing")

if best_coin is None:
    st.info("Geen goede trade kansen gevonden op dit moment.")
else:
    st.write(f"Beste trade kans: **{best_signal.upper()}** {best_coin} @ €{best_price:.2f}")

    if best_signal == "buy" and st.session_state.balance >= best_price:
        st.session_state.open_positions.append({'coin': best_coin, 'entry_price': best_price})
        st.session_state.balance -= best_price
        st.success(f"✅ Gekocht {best_coin} voor €{best_price:.2f}")

    elif best_signal == "sell":
        held_positions = [pos for pos in st.session_state.open_positions if pos['coin'] == best_coin]
        if held_positions:
            position = held_positions[0]
            profit = best_price - position['entry_price']
            st.session_state.balance += best_price
            st.session_state.open_positions.remove(position)
            st.session_state.trade_history.append({
                'coin': best_coin,
                'entry': position['entry_price'],
                'exit': best_price,
                'profit': profit,
            })
            st.success(f"🛒 Verkocht {best_coin} voor €{best_price:.2f} | Winst: €{profit:.2f}")
        else:
            st.warning("Je hebt deze coin niet om te verkopen.")

st.subheader("💰 Huidige balans")
st.write(f"€{st.session_state.balance:.2f}")

st.subheader("📌 Open posities")
if st.session_state.open_positions:
    st.dataframe(pd.DataFrame(st.session_state.open_positions))
else:
    st.info("Geen open posities.")

st.subheader("📈 Trade geschiedenis")
if st.session_state.trade_history:
    df_history = pd.DataFrame(st.session_state.trade_history)
    st.dataframe(df_history)
    totaal_winst = df_history['profit'].sum()
    st.metric("Totale winst (paper trading)", f"€{totaal_winst:.2f}")
else:
    st.info("Geen afgesloten trades.")
