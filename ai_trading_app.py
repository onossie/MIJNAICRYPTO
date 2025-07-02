import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from python_bitvavo_api.bitvavo import Bitvavo
import time

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
balance = START_BALANCE
paper_trading = True

open_positions = []  # lijst met dicts: {'coin', 'entry_price'}
trade_history = []   # lijst met dicts: {'coin', 'entry', 'exit', 'profit', 'accuracy'}

MIN_CANDLES = 50
MAX_COINS = 100  # Voor performance beperken; kan omhoog indien gewenst

# ====== HELPERS =======
@st.cache_data(ttl=3600)
def get_all_eur_markets():
    markets = bitvavo.markets()
    coins = [m['market'] for m in markets if m['quote'] == 'EUR']
    return coins[:MAX_COINS]

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

def predict_action(model, scaler, latest_row):
    X = latest_row[["open", "high", "low", "close", "volume"]]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[-1]
    return pred  # 1 = koop, 0 = verkoop

# ====== STREAMLIT UI =======
st.title("🤖 AI Crypto Trading Bot — Self-learning met budget €96")
st.markdown("Paper trading modus actief — het startbudget is €96.")

coin_list = get_all_eur_markets()

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

# ====== Trading loop =======
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

    model, scaler, accuracy = train_model(df)
    if model is None:
        continue

    latest = df.tail(1)
    decision = predict_action(model, scaler, latest)

    current_price = latest['close'].values[0]

    # Zoek de meest betrouwbare koop- of verkoopkans
    if decision == 1 and accuracy > best_accuracy and st.session_state.balance >= current_price:
        best_coin = coin
        best_signal = "buy"
        best_price = current_price
        best_accuracy = accuracy

    elif decision == 0 and accuracy > best_accuracy:
        # We kunnen alleen verkopen als we de coin al hebben
        held_positions = [pos for pos in st.session_state.open_positions if pos['coin'] == coin]
        if held_positions:
            best_coin = coin
            best_signal = "sell"
            best_price = current_price
            best_accuracy = accuracy

    progress_bar.progress((i+1)/len(coin_list))
    time.sleep(0.05)  # iets vertraging voor UI smoothness

progress_text.text("Klaar met analyseren.")

# ====== Execute trade =======
st.subheader("📊 Trade Beslissing")

if best_coin is None:
    st.info("Geen goede trade kansen gevonden op dit moment.")
else:
    st.write(f"Beste trade kans: **{best_signal.upper()}** {best_coin} @ €{best_price:.2f} (Model acc: {best_accuracy:.2f})")

    if best_signal == "buy" and st_
