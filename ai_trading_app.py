import streamlit as st
from streamlit_autorefresh import st_autorefresh
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
import concurrent.futures
import threading
import websocket
import json
import matplotlib.pyplot as plt

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
MIN_CANDLES = 50
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TAKE_PROFIT_PCT = 0.02
STOP_LOSS_PCT = 0.015
TRAILING_STOP_TRIGGER_PCT = 0.015
TRAILING_STOP_GAP_PCT = 0.005

# ====== Auto refresh elke 10 seconden =======
st_autorefresh(interval=10_000, key="auto_refresh")

# ====== Helper functies =======
def get_all_eur_markets():
    markets = bitvavo.markets()
    return [m['market'] for m in markets if m['quote'] == 'EUR']

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
    if model and scaler:
        return model, scaler
    model, scaler, _ = train_model(df)
    if model and scaler:
        save_model_and_scaler(coin, model, scaler)
    return model, scaler

def predict_action(model, scaler, latest_row):
    X = latest_row[["open", "high", "low", "close", "volume"]]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[-1]
    return pred  # 1 = buy, 0 = sell

def analyze_coin(coin):
    df = get_historical_prices(coin)
    if df is None:
        return None
    model, scaler = train_or_load_model(df, coin)
    if model is None or scaler is None:
        return None
    latest = df.tail(1)
    decision = predict_action(model, scaler, latest)
    current_price = latest['close'].values[0]
    accuracy = None
    try:
        _, _, accuracy = train_model(df)
    except:
        accuracy = None
    return {
        'coin': coin,
        'decision': decision,
        'price': current_price,
        'accuracy': accuracy,
        'df': df
    }

# ====== WebSocket live prices =======
def on_message(ws, message):
    data = json.loads(message)
    if "market" in data and "close" in data:
        coin = data["market"]
        price = float(data["close"])
        if 'live_prices' not in st.session_state:
            st.session_state.live_prices = {}
        st.session_state.live_prices[coin] = price

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    # Subscribe to ticker for all EUR markets
    markets = get_all_eur_markets()
    params = [{"market": m, "interval": "1m"} for m in markets]
    subscribe_msg = {
        "action": "subscribe",
        "subscriptions": [
            {"name": "ticker", "markets": markets}
        ]
    }
    ws.send(json.dumps(subscribe_msg))

def start_ws_client():
    ws_url = "wss://ws.bitvavo.com/v2/"
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever()

# Start WebSocket thread once
if 'ws_thread_started' not in st.session_state:
    threading.Thread(target=start_ws_client, daemon=True).start()
    st.session_state.ws_thread_started = True

# ====== Streamlit UI =======
st.title("🤖 AI Crypto Trading Bot - Realtime met WebSocket prijzen")
st.markdown("Paper trading modus actief — startbudget is €96.")

if 'balance' not in st.session_state:
    st.session_state.balance = START_BALANCE

if 'open_positions' not in st.session_state:
    st.session_state.open_positions = []

if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

if 'live_prices' not in st.session_state:
    st.session_state.live_prices = {}

st.sidebar.header("⚙️ Handmatige controls")
if st.sidebar.button("🔄 Reset paper trading"):
    st.session_state.balance = START_BALANCE
    st.session_state.open_positions = []
    st.session_state.trade_history = []
    st.session_state.live_prices = {}

# ====== Analyseer coins parallel =======
coin_list = get_all_eur_markets()
st.subheader(f"Analyseer {len(coin_list)} EUR coins in parallel...")

progress_text = st.empty()
progress_bar = st.progress(0)

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(analyze_coin, coin): coin for coin in coin_list}
    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        result = future.result()
        if result:
            results.append(result)
        progress_text.text(f"Analyseer coins: {i+1}/{len(coin_list)}")
        progress_bar.progress((i+1)/len(coin_list))
progress_text.text("Analyse voltooid.")

# ====== Zoek beste trade kans =======
best_coin = None
best_signal = None
best_price = None

# Koop signaal als er genoeg saldo is
for res in results:
    if res['decision'] == 1 and st.session_state.balance >= res['price']:
        best_coin = res['coin']
        best_signal = "buy"
        best_price = res['price']
        break

# Check verkoopsignalen voor open posities
if best_coin is None:
    for pos in st.session_state.open_positions:
        coin = pos['coin']
        found = next((r for r in results if r['coin'] == coin), None)
        if found:
            best_coin = coin
            best_signal = "check_exit"
            best_price = found['price']
            break

# ====== Trade logica =======
st.subheader("📊 Trade Beslissing")

if best_coin is None:
    st.info("Geen goede trade kansen gevonden op dit moment.")
else:
    st.write(f"Beslissing: **{best_signal.upper()}** {best_coin} @ €{best_price:.4f}")

    if best_signal == "buy" and st.session_state.balance >= best_price:
        st.session_state.open_positions.append({
            'coin': best_coin,
            'entry_price': best_price,
            'highest_price': best_price
        })
        st.session_state.balance -= best_price
        st.success(f"✅ Gekocht {best_coin} voor €{best_price:.4f}")

    elif best_signal == "check_exit":
        for position in st.session_state.open_positions:
            if position['coin'] == best_coin:
                entry = position['entry_price']
                highest = position.get('highest_price', best_price)
                position['highest_price'] = max(highest, best_price)
                change_pct = (best_price - entry) / entry

                reason = None
                if change_pct >= TAKE_PROFIT_PCT:
                    reason = "🎯 Take profit"
                elif change_pct <= -STOP_LOSS_PCT:
                    reason = "🛑 Stop loss"
                elif change_pct >= TRAILING_STOP_TRIGGER_PCT:
                    if best_price <= position['highest_price'] * (1 - TRAILING_STOP_GAP_PCT):
                        reason = "🔃 Trailing stop"

                if reason:
                    profit = best_price - entry
                    st.session_state.balance += best_price
                    st.session_state.open_positions.remove(position)
                    st.session_state.trade_history.append({
                        'coin': best_coin,
                        'entry': entry,
                        'exit': best_price,
                        'profit': profit,
                        'reason': reason
                    })
                    st.warning(f"Verkocht {best_coin} voor €{best_price:.4f} ({reason})")
                break

st.markdown("---")
st.subheader("💼 Open posities — realtime prijzen")

if st.session_state.open_positions:
    df_pos = pd.DataFrame(st.session_state.open_positions)
    df_pos['current_price'] = df_pos['coin'].apply(lambda c: st.session_state.live_prices.get(c, np.nan))
    df_pos['unrealized_profit'] = df_pos['current_price'] - df_pos['entry_price']
    df_pos['unrealized_profit_pct'] = (df_pos['unrealized_profit'] / df_pos['entry_price']) * 100

    def highlight(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
        return f'color: {color}'

    st.dataframe(
        df_pos[['coin', 'entry_price', 'current_price', 'unrealized_profit', 'unrealized_profit_pct']].style
        .format({
            'entry_price': '€{:.4f}',
            'current_price': '€{:.4f}',
            'unrealized_profit': '€{:.4f}',
            'unrealized_profit_pct': '{:.2f}%'
        })
        .applymap(highlight, subset=['unrealized_profit', 'unrealized_profit_pct'])
    )
    st.bar_chart(df_pos.set_index("coin")["unrealized_profit"])
else:
    st.info("📭 Geen open posities.")

st.markdown("---")
st.subheader("💰 Beschikbaar saldo")
st.metric("€ EUR saldo", f"€{st.session_state.balance:.2f}")

st.markdown("---")
st.subheader("📜 Trade geschiedenis")
if st.session_state.trade_history:
    df_hist = pd.DataFrame(st.session_state.trade_history)
    st.dataframe(df_hist.style.format({
        'entry': '€{:.4f}',
        'exit': '€{:.4f}',
        'profit': '€{:.4f}'
    }))
else:
    st.info("Geen trades gemaakt.")

# Extra: toon accuracies van modellen (indien beschikbaar)
accuracies = [r['accuracy'] for r in results if r['accuracy'] is not None]
if accuracies:
    avg_acc = np.mean(accuracies)
    st.write(f"Gemiddelde model accuracy (laatste training): {avg_acc:.2f}")
else:
    st.write("Geen accuracy data beschikbaar.")
