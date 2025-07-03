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

# ====== AUTO REFRESH INSTELLING ======
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

# ====== Streamlit UI =======
st.title("🤖 AI Crypto Trading Bot — Autorefresh elke 60 seconden")
st.markdown("Paper trading modus actief — het startbudget is €96.")

# Init session state
if 'balance' not in st.session_state:
    st.session_state.balance = START_BALANCE

if 'open_positions' not in st.session_state:
    st.session_state.open_positions = []

if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

if 'balance_history' not in st.session_state:
    st.session_state.balance_history = [START_BALANCE]

if 'accuracy_per_coin' not in st.session_state:
    st.session_state.accuracy_per_coin = {}

# Sidebar controls
st.sidebar.header("⚙️ Handmatige controls")
if st.sidebar.button("🔄 Reset paper trading"):
    st.session_state.balance = START_BALANCE
    st.session_state.open_positions = []
    st.session_state.trade_history = []
    st.session_state.balance_history = [START_BALANCE]
    st.session_state.accuracy_per_coin = {}

# ====== AI Analyse en Trade Logic =======
coin_list = get_all_eur_markets()
st.subheader("📈 AI Trading Actie")

valid_coins = []
coin_data = {}

# Stap 1: Bepaal geldige coins (voldoende candles + model mogelijk)
for coin in coin_list:
    df = get_historical_prices(coin)
    if df is None:
        continue
    model, scaler = load_model_and_scaler(coin)
    if model is None or scaler is None:
        model, scaler, accuracy = train_model(df)
        if model is None:
            continue
        save_model_and_scaler(coin, model, scaler)
        st.session_state.accuracy_per_coin[coin] = accuracy
    else:
        accuracy = st.session_state.accuracy_per_coin.get(coin, None)

    valid_coins.append(coin)
    coin_data[coin] = {
        'df': df,
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy
    }

if not valid_coins:
    st.warning("Geen geldige coins gevonden om mee te handelen.")
else:
    per_trade_budget = st.session_state.balance / len(valid_coins) if len(valid_coins) > 0 else 0
    progress_bar = st.progress(0)
    trades_today = 0

    for i, coin in enumerate(valid_coins):
        progress_bar.progress((i + 1) / len(valid_coins))
        df = coin_data[coin]['df']
        model = coin_data[coin]['model']
        scaler = coin_data[coin]['scaler']
        latest = df.tail(1)
        decision = predict_action(model, scaler, latest)
        current_price = latest['close'].values[0]

        # Check of we al deze coin bezitten
        already_holding = any(pos['coin'] == coin for pos in st.session_state.open_positions)

        # Buy logic
        if decision == 1 and st.session_state.balance >= current_price and not already_holding:
            st.session_state.open_positions.append({
                'coin': coin,
                'entry_price': current_price,
                'highest_price': current_price
            })
            st.session_state.balance -= current_price
            trades_today += 1
            st.write(f"✅ Gekocht {coin} voor €{current_price:.2f}")

        # Sell logic
        elif decision == 0 and already_holding:
            for position in st.session_state.open_positions:
                if position['coin'] == coin:
                    entry = position['entry_price']
                    highest = position.get('highest_price', current_price)
                    position['highest_price'] = max(highest, current_price)
                    change_pct = (current_price - entry) / entry

                    reason = None
                    if change_pct >= TAKE_PROFIT_PCT:
                        reason = "🎯 Take profit"
                    elif change_pct <= -STOP_LOSS_PCT:
                        reason = "🛑 Stop loss"
                    elif change_pct >= TRAILING_STOP_TRIGGER_PCT and current_price <= highest * (1 - TRAILING_STOP_GAP_PCT):
                        reason = "🔃 Trailing stop"

                    if reason:
                        profit = current_price - entry
                        st.session_state.balance += current_price
                        st.session_state.open_positions.remove(position)
                        st.session_state.trade_history.append({
                            'coin': coin,
                            'entry': entry,
                            'exit': current_price,
                            'profit': profit,
                            'reason': reason
                        })
                        st.success(f"{reason} - Verkocht {coin} @ €{current_price:.2f} | Winst: €{profit:.2f}")
                    break
        time.sleep(0.05)

    progress_bar.empty()
    st.info(f"Analyse compleet. {trades_today} nieuwe kooporders geplaatst.")

# Update balans geschiedenis
if len(st.session_state.balance_history) == 0 or st.session_state.balance_history[-1] != st.session_state.balance:
    st.session_state.balance_history.append(st.session_state.balance)

# ====== Resultaten tonen =======
st.subheader("💰 Huidige balans")
st.write(f"€{st.session_state.balance:.2f}")

st.subheader("📌 Open posities")
if st.session_state.open_positions:
    df_positions = pd.DataFrame(st.session_state.open_positions)
    # Haal actuele prijzen erbij voor grafiek
    current_prices = []
    for pos in st.session_state.open_positions:
        df = get_historical_prices(pos['coin'])
        if df is not None:
            current_prices.append(df['close'].iloc[-1])
        else:
            current_prices.append(np.nan)
    df_positions['current_price'] = current_prices
    st.dataframe(df_positions)
else:
    st.info("Geen open posities.")

st.subheader("📈 Trade geschiedenis")
if st.session_state.trade_history:
    df_history = pd.DataFrame(st.session_state.trade_history)
    st.dataframe(df_history)
    totaal_winst = df_history['profit'].sum()
    st.metric("Totale winst (paper trading)", f"€{totaal_winst:.2f}")
else:
    st.info("Nog geen trades afgesloten.")

# ====== Visualisaties =======
st.subheader("📊 Visualisaties")

# 1. Balans over tijd
fig1, ax1 = plt.subplots()
ax1.plot(st.session_state.balance_history, marker='o')
ax1.set_title("Balans over tijd")
ax1.set_xlabel("Update #")
ax1.set_ylabel("€")
st.pyplot(fig1)

# 2. Winst per coin (uit trade history)
if st.session_state.trade_history:
    df_hist = pd.DataFrame(st.session_state.trade_history)
    winst_per_coin = df_hist.groupby('coin')['profit'].sum().sort_values(ascending=False)
    fig2, ax2 = plt.subplots()
    winst_per_coin.plot(kind='bar', ax=ax2, color='green')
    ax2.set_title("Totale winst per coin")
    ax2.set_ylabel("€ winst")
    st.pyplot(fig2)
else:
    st.write("Nog geen winst per coin beschikbaar.")

# 3. Accuracy per coin
if st.session_state.accuracy_per_coin:
    acc_df = pd.DataFrame.from_dict(st.session_state.accuracy_per_coin, orient='index', columns=['accuracy'])
    acc_df_sorted = acc_df.sort_values('accuracy', ascending=False)
    fig3, ax3 = plt.subplots()
    acc_df_sorted['accuracy'].plot(kind='bar', ax=ax3, color='blue')
    ax3.set_title("Model accuracy per coin")
    ax3.set_ylabel("Accuracy")
    ax3.set_ylim(0,1)
    st.pyplot(fig3)
else:
    st.write("Nog geen model accuracy beschikbaar.")

# 4. Open posities prijsvergelijking
if st.session_state.open_positions:
    fig4, ax4 = plt.subplots()
    ax4.bar(df_positions['coin'], df_positions['current_price'], label='Huidige prijs')
    ax4.bar(df_positions['coin'], df_positions['entry_price'], label='Instapprijs', alpha=0.7)
    ax4.set_title("Open posities: Instapprijs vs huidige prijs")
    ax4.set_ylabel("€")
    ax4.legend()
    st.pyplot(fig4)
else:
    st.write("Geen open posities om te visualiseren.")
