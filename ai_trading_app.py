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
MIN_CANDLES = 50
MIN_PRICE = 0.50  # Filter voor te goedkope coins

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ====== Cache functies =======
@st.cache_data(ttl=3600)
def get_all_eur_markets():
    markets = bitvavo.markets()
    coins = [m['market'] for m in markets if m['quote'] == 'EUR']
    return coins

@st.cache_data(ttl=300)
def get_historical_prices(symbol, interval='1h', limit=100):
    try:
        candles = bitvavo.candles(symbol, interval, {"limit": limit})
        if len(candles) < MIN_CANDLES:
            return None
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        return None

# ====== Model functies =======
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
        # We laden accuracy niet op, dus altijd opnieuw trainen om accuracy te krijgen
        model, scaler, accuracy = train_model(df)
        if model and scaler:
            save_model_and_scaler(coin, model, scaler)
        return model, scaler, accuracy
    else:
        model, scaler, accuracy = train_model(df)
        if model and scaler:
            save_model_and_scaler(coin, model, scaler)
        return model, scaler, accuracy

def predict_action(model, scaler, latest_row, threshold=0.55):
    X = latest_row[["open", "high", "low", "close", "volume"]]
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[-1]
    if prob[1] > threshold:
        return 1  # Buy
    elif prob[0] > threshold:
        return 0  # Sell
    else:
        return -1  # Hold

# ====== Streamlit UI =======
st.title("🤖 Verbeterde AI Crypto Trading Bot — Paper trading met €96 startkapitaal")

st.markdown("Paper trading modus actief. De bot kijkt naar alle EUR-coins op Bitvavo en kiest de beste trade op basis van zelflerende modellen.")

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

coins = get_all_eur_markets()
st.write(f"Aantal EUR-coins gevonden: {len(coins)}")

progress_text = st.empty()
progress_bar = st.progress(0)

best_coin = None
best_signal = None
best_price = None
best_accuracy = 0.0

trained_models = 0
skipped_due_to_accuracy = 0
skipped_due_to_price = 0

for i, coin in enumerate(coins):
    progress_text.text(f"Analyseer {coin} ({i+1}/{len(coins)})")
    df = get_historical_prices(coin)
    if df is None:
        continue

    current_price = df['close'].iloc[-1]
    if current_price < MIN_PRICE:
        skipped_due_to_price += 1
        continue

    model, scaler, accuracy = train_or_load_model(df, coin)
    if model is None or scaler is None or accuracy is None:
        continue

    if accuracy < 0.55:
        skipped_due_to_accuracy += 1
        continue

    trained_models += 1

    latest = df.tail(1)
    decision = predict_action(model, scaler, latest)

    if decision == 1 and st.session_state.balance >= current_price and accuracy > best_accuracy:
        best_coin = coin
        best_signal = "buy"
        best_price = current_price
        best_accuracy = accuracy

    elif decision == 0:
        held_positions = [pos for pos in st.session_state.open_positions if pos['coin'] == coin]
        if held_positions and accuracy > best_accuracy:
            best_coin = coin
            best_signal = "sell"
            best_price = current_price
            best_accuracy = accuracy

    progress_bar.progress((i + 1) / len(coins))
    time.sleep(0.05)

progress_text.text("Analyse klaar.")

st.write(f"Modellen getraind: {trained_models}")
st.write(f"Coins overgeslagen wegens lage accuracy: {skipped_due_to_accuracy}")
st.write(f"Coins overgeslagen wegens lage prijs (<€{MIN_PRICE}): {skipped_due_to_price}")

st.subheader("📊 Beste trade beslissing")
if best_coin is None:
    st.info("Geen goede trade kansen gevonden op dit moment.")
else:
    st.write(f"Beste trade: **{best_signal.upper()}** {best_coin} @ €{best_price:.4f} (Accuracy: {best_accuracy:.2f})")

    if best_signal == "buy" and st.session_state.balance >= best_price:
        st.session_state.open_positions.append({'coin': best_coin, 'entry_price': best_price})
        st.session_state.balance -= best_price
        st.success(f"✅ Gekocht {best_coin} voor €{best_price:.4f}")

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
            st.success(f"🛒 Verkocht {best_coin} voor €{best_price:.4f} | Winst: €{profit:.4f}")
        else:
            st.warning("Je hebt deze coin niet om te verkopen.")

st.subheader("💰 Huidige balans")
st.write(f"€{st.session_state.balance:.2f}")

st.subheader("📌 Open posities")
if st.session_state.open_positions:
    df_positions = pd.DataFrame(st.session_state.open_positions)
    # Voeg huidige prijs en winst toe
    current_prices = []
    profits = []
    for pos in st.session_state.open_positions:
        df_coin = get_historical_prices(pos['coin'], limit=1)
        price = df_coin['close'].iloc[-1] if df_coin is not None else np.nan
        current_prices.append(price)
        profits.append(price - pos['entry_price'] if price is not np.nan else np.nan)
    df_positions['current_price'] = current_prices
    df_positions['profit'] = profits
    st.dataframe(df_positions)
else:
    st.info("Geen open posities.")

st.subheader("📈 Trade geschiedenis")
if st.session_state.trade_history:
    df_history = pd.DataFrame(st.session_state.trade_history)
    st.dataframe(df_history)
    totaal_winst = df_history['profit'].sum()
    st.metric("Totale winst (paper trading)", f"€{totaal_winst:.2f}")

    # Winst grafiek
    df_history['cumulative_profit'] = df_history['profit'].cumsum()
    st.line_chart(df_history.set_index(df_history.index)['cumulative_profit'])
else:
    st.info("Geen afgesloten trades.")
