import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import os
import time
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from python_bitvavo_api.bitvavo import Bitvavo

# ====== AUTO REFRESH ======
st_autorefresh(interval=60000, limit=None, key="auto-refresh")

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

# Trade instellingen
TAKE_PROFIT_PCT = 0.02
STOP_LOSS_PCT = 0.015
TRAILING_STOP_TRIGGER_PCT = 0.015
TRAILING_STOP_GAP_PCT = 0.005

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

# ====== INITIALISEER SESSION STATE =======
if 'balance' not in st.session_state:
    st.session_state.balance = START_BALANCE

if 'open_positions' not in st.session_state:
    st.session_state.open_positions = []

if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

if 'equity_curve' not in st.session_state:
    st.session_state.equity_curve = [(pd.Timestamp.now(), START_BALANCE)]

# ====== UI =======
st.title("🤖 AI Crypto Trading Bot — Automatische Visualisatie")
st.markdown("Paper trading modus actief — startbudget is €96.")

st.sidebar.header("⚙️ Handmatige controls")
if st.sidebar.button("🔄 Reset paper trading"):
    st.session_state.balance = START_BALANCE
    st.session_state.open_positions = []
    st.session_state.trade_history = []
    st.session_state.equity_curve = [(pd.Timestamp.now(), START_BALANCE)]

# ====== AI Analyse en Trading =======
coin_list = get_all_eur_markets()
st.subheader("📈 AI Trading Acties")

best_coin = None
best_signal = None
best_price = None

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

    if decision == 1 and st.session_state.balance >= current_price:
        best_coin = coin
        best_signal = "buy"
        best_price = current_price
        break
    elif decision == 0:
        for position in st.session_state.open_positions:
            if position['coin'] == coin:
                best_coin = coin
                best_signal = "check_exit"
                best_price = current_price
                break

    progress_bar.progress((i + 1) / len(coin_list))
    time.sleep(0.01)

progress_text.text("Klaar met analyseren.")

# ====== Trade Uitvoering =======
st.subheader("📊 Trade Beslissing")

if best_coin is None:
    st.info("Geen goede trade kansen gevonden op dit moment.")
else:
    st.write(f"Beslissing: **{best_signal.upper()}** {best_coin} @ €{best_price:.2f}")

    if best_signal == "buy" and st.session_state.balance >= best_price:
        st.session_state.open_positions.append({
            'coin': best_coin,
            'entry_price': best_price,
            'highest_price': best_price,
            'entry_time': pd.Timestamp.now()
        })
        st.session_state.balance -= best_price
        st.success(f"✅ Gekocht {best_coin} voor €{best_price:.2f}")

    elif best_signal == "check_exit":
        for position in st.session_state.open_positions:
            if position['coin'] == best_coin:
                entry = position['entry_price']
                highest = position.get('highest_price', best_price)
                position['highest_price'] = max(highest, best_price)
                change_pct = (best_price - entry) / entry

                exit_reason = None
                if change_pct >= TAKE_PROFIT_PCT:
                    exit_reason = "🎯 Take profit"
                elif change_pct <= -STOP_LOSS_PCT:
                    exit_reason = "🛑 Stop loss"
                elif change_pct >= TRAILING_STOP_TRIGGER_PCT:
                    if best_price <= position['highest_price'] * (1 - TRAILING_STOP_GAP_PCT):
                        exit_reason = "🔃 Trailing stop"

                if exit_reason:
                    profit = best_price - entry
                    st.session_state.balance += best_price
                    st.session_state.open_positions.remove(position)
                    st.session_state.trade_history.append({
                        'coin': best_coin,
                        'entry': entry,
                        'exit': best_price,
                        'profit': profit,
                        'reason': exit_reason,
                        'entry_time': position['entry_time'],
                        'exit_time': pd.Timestamp.now()
                    })
                    st.success(f"{exit_reason} - Verkocht {best_coin} @ €{best_price:.2f} | Winst: €{profit:.2f}")
                    break

# ====== Update equity curve =======
latest_equity = st.session_state.equity_curve[-1][1]
# Equity = balans + open posities waarde
open_positions_value = 0
for pos in st.session_state.open_positions:
    # Probeer huidige prijs op te halen
    df_pos = get_historical_prices(pos['coin'], limit=1)
    if df_pos is not None:
        current_close = df_pos['close'].values[-1]
        open_positions_value += current_close
equity = st.session_state.balance + open_positions_value
st.session_state.equity_curve.append((pd.Timestamp.now(), equity))

# ====== Visualisaties =======
st.subheader("📈 Equity Curve (Balans + Open Posities)")

# Equity curve plot
times, balances = zip(*st.session_state.equity_curve)
fig, ax = plt.subplots()
ax.plot(times, balances, marker='o')
ax.set_xlabel("Tijd")
ax.set_ylabel("Equity (€)")
ax.set_title("Equity Curve")
ax.grid(True)
fig.autofmt_xdate()
st.pyplot(fig)

st.subheader("💰 Huidige balans en open posities")
st.write(f"€{st.session_state.balance:.2f} beschikbaar")

if st.session_state.open_positions:
    st.dataframe(pd.DataFrame(st.session_state.open_positions))
else:
    st.info("Geen open posities.")

st.subheader("📊 Trade geschiedenis")
if st.session_state.trade_history:
    df_history = pd.DataFrame(st.session_state.trade_history)
    st.dataframe(df_history)
    totaal_winst = df_history['profit'].sum()
    st.metric("Totale winst (paper trading)", f"€{totaal_winst:.2f}")

    # Plot trades op prijsgrafiek van laatste coin in trade_history
    laatste_trade = df_history.iloc[-1]
    coin = laatste_trade['coin']
    df_coin = get_historical_prices(coin, limit=100)
    if df_coin is not None:
        fig2, ax2 = plt.subplots()
        ax2.plot(df_coin['timestamp'], df_coin['close'], label='Prijs (€)')
        ax2.axhline(y=laatste_trade['entry'], color='green', linestyle='--', label='Entry prijs')
        ax2.axhline(y=laatste_trade['exit'], color='red', linestyle='--', label='Exit prijs')
        ax2.set_title(f"Trade prijs grafiek {coin}")
        ax2.set_xlabel("Tijd")
        ax2.set_ylabel("Prijs (€)")
        ax2.legend()
        fig2.autofmt_xdate()
        st.pyplot(fig2)
else:
    st.info("Nog geen afgesloten trades.")

