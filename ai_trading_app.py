import streamlit as st
import threading
import time
import queue
import json
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from python_bitvavo_api.bitvavo import Bitvavo
import matplotlib.pyplot as plt

# ===== CONFIG =====
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

# ===== Queue en thread control =====
ws_data_queue = queue.Queue()
stop_ws_thread = False

# ===== Helper functies =====
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

# ===== WebSocket callback functies =====
def on_message(ws, message):
    msg = json.loads(message)
    if "market" in msg and "candles" not in msg:
        # Dit is een trade of ticker update; we willen candle updates
        return
    if isinstance(msg, list):
        # candle update bericht (Bitvavo WS stuurt list van candles)
        # stuur naar queue zodat main thread het verwerkt
        for m in msg:
            if "market" in m:
                ws_data_queue.put(m)

def on_error(ws, error):
    print("WS Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket gesloten")

def on_open(ws):
    # Subscribe op alle EUR candles 1h interval (let op max aantal abonnees Bitvavo)
    eur_markets = get_all_eur_markets()
    params = [{"market": m, "interval": "1h"} for m in eur_markets]
    subscribe_msg = {
        "action": "subscribe",
        "subscriptions": [
            {
                "name": "candles",
                "markets": eur_markets,
                "interval": "1h"
            }
        ]
    }
    ws.send(json.dumps(subscribe_msg))

# ===== WebSocket runner thread =====
def run_ws():
    global stop_ws_thread
    from websocket import WebSocketApp
    while not stop_ws_thread:
        try:
            ws = WebSocketApp(
                "wss://ws.bitvavo.com/v2/",
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            ws.run_forever()
        except Exception as e:
            print(f"WS fout, reconnect over 5s: {e}")
            time.sleep(5)

# ===== Main Streamlit app =====
def main():
    st.title("🤖 Realtime AI Crypto Trading Bot met WebSocket")

    # Initialiseer sessiestate
    if 'balance' not in st.session_state:
        st.session_state.balance = START_BALANCE
    if 'open_positions' not in st.session_state:
        st.session_state.open_positions = []
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = []
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'scalers' not in st.session_state:
        st.session_state.scalers = {}
    if 'latest_prices' not in st.session_state:
        st.session_state.latest_prices = {}
    if 'ws_thread_started' not in st.session_state:
        st.session_state.ws_thread_started = False
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = {}

    # Start WebSocket thread (één keer)
    if not st.session_state.ws_thread_started:
        import threading
        ws_thread = threading.Thread(target=run_ws, daemon=True)
        ws_thread.start()
        st.session_state.ws_thread_started = True

    st.sidebar.header("⚙️ Handmatige controls")
    if st.sidebar.button("🔄 Reset paper trading"):
        st.session_state.balance = START_BALANCE
        st.session_state.open_positions = []
        st.session_state.trade_history = []

    # Verwerk inkomende WS data uit queue
    new_data = False
    while not ws_data_queue.empty():
        candle_update = ws_data_queue.get()
        market = candle_update['market']
        # Update laatste prijs
        close_price = float(candle_update['close'])
        st.session_state.latest_prices[market] = close_price

        # Opslaan candle in sessiestate voor analyse
        if market not in st.session_state.last_analysis:
            st.session_state.last_analysis[market] = []
        # Voeg candle toe (timestamp, open, high, low, close, volume)
        st.session_state.last_analysis[market].append({
            "timestamp": pd.to_datetime(candle_update['start']),
            "open": float(candle_update['open']),
            "high": float(candle_update['high']),
            "low": float(candle_update['low']),
            "close": close_price,
            "volume": float(candle_update['volume'])
        })
        # Hou max 200 candles per coin (memory limit)
        if len(st.session_state.last_analysis[market]) > 200:
            st.session_state.last_analysis[market].pop(0)
        new_data = True

    # Als er nieuwe data is, voer AI analyse uit (slim throttling)
    if new_data:
        # Voor performance: analyseer alleen coins met minimaal MIN_CANDLES candles
        for coin, candles in st.session_state.last_analysis.items():
            if len(candles) < MIN_CANDLES:
                continue
            # Zet candles om naar DataFrame
            df = pd.DataFrame(candles)
            # Train of laad model
            if coin not in st.session_state.models:
                model, scaler = train_or_load_model(df, coin)
                if model and scaler:
                    st.session_state.models[coin] = model
                    st.session_state.scalers[coin] = scaler
            else:
                model = st.session_state.models[coin]
                scaler = st.session_state.scalers[coin]

            if model is None or scaler is None:
                continue

            # Voorspel actie op laatste candle
            latest_candle = df.tail(1)
            action = predict_action(model, scaler, latest_candle)

            # Handel simulatie paper trading
            price = st.session_state.latest_prices.get(coin, None)
            if price is None:
                continue

            # Check open positie
            open_pos = next((pos for pos in st.session_state.open_positions if pos['coin'] == coin), None)

            if action == 1 and open_pos is None and st.session_state.balance > 1:
                # Koop
                qty = st.session_state.balance / price
                st.session_state.balance -= qty * price
                st.session_state.open_positions.append({
                    "coin": coin,
                    "qty": qty,
                    "entry_price": price,
                    "max_price": price,
                    "entry_time": time.time()
                })
                st.session_state.trade_history.append({
                    "type": "buy",
                    "coin": coin,
                    "price": price,
                    "qty": qty,
                    "time": time.time()
                })

            elif action == 0 and open_pos is not None:
                # Verkoop
                qty = open_pos['qty']
                st.session_state.balance += qty * price
                st.session_state.open_positions.remove(open_pos)
                st.session_state.trade_history.append({
                    "type": "sell",
                    "coin": coin,
                    "price": price,
                    "qty": qty,
                    "time": time.time()
                })

            # Update trailing stop
            if open_pos:
                if price > open_pos['max_price']:
                    open_pos['max_price'] = price
                trailing_stop = open_pos['max_price'] * (1 - TRAILING_STOP_GAP_PCT)
                if price < trailing_stop:
                    # Trailing stop triggered: verkoop
                    qty = open_pos['qty']
                    st.session_state.balance += qty * price
                    st.session_state.open_positions.remove(open_pos)
                    st.session_state.trade_history.append({
                        "type": "sell",
                        "coin": coin,
                        "price": price,
                        "qty": qty,
                        "time": time.time()
                    })

    # UI: Toon balance & open posities
    st.header("📊 Paper Trading Balans")
    st.write(f"Start balans: €{START_BALANCE:.2f}")
    st.write(f"Huidige balans: €{st.session_state.balance:.2f}")
    st.write(f"Open posities: {len(st.session_state.open_positions)}")

    if st.session_state.open_positions:
        df_pos = pd.DataFrame(st.session_state.open_positions)
        df_pos["current_price"] = df_pos["coin"].apply(lambda c: st.session_state.latest_prices.get(c, np.nan))
        df_pos["unrealized_pl"] = (df_pos["current_price"] - df_pos["entry_price"]) * df_pos["qty"]
        st.dataframe(df_pos[["coin", "qty", "entry_price", "current_price", "unrealized_pl"]])

    # UI: Toon trade historie
    st.header("📈 Trade Historie (laatste 20 trades)")
    if st.session_state.trade_history:
        df_hist = pd.DataFrame(st.session_state.trade_history)
        df_hist['time'] = pd.to_datetime(df_hist['time'], unit='s')
        st.dataframe(df_hist.tail(20)[["type", "coin", "price", "qty", "time"]])
    else:
        st.write("Geen trades gedaan.")

    # UI: Plot laatste prijs van top coins
    st.header("📉 Laatste Prijzen (Top 10 coins met data)")
    latest_prices = st.session_state.latest_prices
    if latest_prices:
        top10 = dict(sorted(latest_prices.items(), key=lambda item: item[1], reverse=True)[:10])
        fig, ax = plt.subplots()
        ax.bar(top10.keys(), top10.values())
        ax.set_ylabel("Prijs (EUR)")
        st.pyplot(fig)
    else:
        st.write("Nog geen data ontvangen.")

if __name__ == "__main__":
    main()
