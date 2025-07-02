import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from python_bitvavo_api.bitvavo import Bitvavo

# Load secrets from Streamlit
API_KEY = st.secrets["BITVAVO_API_KEY"]
API_SECRET = st.secrets["BITVAVO_API_SECRET"]

bitvavo = Bitvavo({
    'APIKEY': API_KEY,
    'APISECRET': API_SECRET,
    'RESTURL': 'https://api.bitvavo.com/v2',
    'WSURL': 'wss://ws.bitvavo.com/v2/'
})

# Initial values
START_BALANCE = 100.0
paper_trading = True
balance = START_BALANCE
open_positions = []

@st.cache_data
def get_markets():
    return bitvavo.markets()

@st.cache_data
def get_historical_prices():
    candles = bitvavo.candles("ETH-EUR", "1h", {"limit": 100})
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def train_ai_model(df):
    df['future_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    df['target'] = (df['future_close'] > df['close']).astype(int)

    X = df[['open', 'high', 'low', 'close', 'volume']]
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, scaler

def make_decision(model, scaler, latest_data):
    X = latest_data[['open', 'high', 'low', 'close', 'volume']]
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    return prediction[-1]  # 1 = Buy, 0 = Sell

def execute_trade(decision, price):
    global balance, open_positions, paper_trading

    if decision == 1 and balance >= price:
        st.info(f"BUY @ {price:.2f} EUR")
        open_positions.append({"entry": price})
        if paper_trading:
            balance -= price
        else:
            bitvavo.placeOrder("ETH-EUR", {'side': 'buy', 'orderType': 'market', 'amount': str(1)})

    elif decision == 0 and open_positions:
        position = open_positions.pop(0)
        profit = price - position["entry"]
        if paper_trading:
            balance += price
        else:
            bitvavo.placeOrder("ETH-EUR", {'side': 'sell', 'orderType': 'market', 'amount': str(1)})
        st.success(f"SELL @ {price:.2f} EUR | Profit: {profit:.2f} EUR")

def toggle_trading():
    global paper_trading
    paper_trading = not paper_trading

# Streamlit UI
st.title("AI Crypto Trading Bot")

if st.button("Toggle: Paper Trading / Live Trading"):
    toggle_trading()

st.write(f"**Mode**: {'🧪 Pap
