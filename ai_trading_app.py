import streamlit as st
import pandas as pd
import numpy as np
from bitvavo import Bitvavo
import ta
from datetime import datetime

# --- Bitvavo API keys vanuit Streamlit Secrets ---
API_KEY = st.secrets["BITVAVO_API_KEY"]
API_SECRET = st.secrets["BITVAVO_API_SECRET"]

bitvavo = Bitvavo({
    'apiKey': API_KEY,
    'apiSecret': API_SECRET,
    'restTimeOffset': 0,
    'wsTimeOffset': 0,
    'demo': False  # Echte trades; verander naar True voor paper trading
})

@st.cache_data(ttl=600)
def get_markets():
    markets = bitvavo.markets()
    eur_markets = [m['market'] for m in markets if m['quote'] == 'EUR' and m['status'] == 'trading']
    return eur_markets

def get_candles(market, interval="1h", limit=200):
    candles = bitvavo.candles(market, interval, {"limit": limit})
    df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit='ms')
    return df

def add_indicators(df):
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_h"] = bb.bollinger_hband()
    df["bb_l"] = bb.bollinger_lband()
    return df

def init_portfolio(markets):
    portfolio = {"EUR": 96.0}  # Start saldo 96 EUR
    for m in markets:
        base = m.split("-")[0]
        portfolio[base] = 0.0
    return portfolio

def init_positions():
    return {}

class QLearningAgent:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        for state in states:
            self.q_table[state] = {a: 0.0 for a in actions}

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            q_vals = self.q_table[state]
            max_q = max(q_vals.values())
            max_actions = [a for a, q in q_vals.items() if q == max_q]
            return np.random.choice(max_actions)

    def update(self, state, action, reward, next_state):
        old_q = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.q_table[state][action] = new_q

def discretize_state(rsi, macd_diff):
    if rsi < 30:
        rsi_state = "low"
    elif rsi > 70:
        rsi_state = "high"
    else:
        rsi_state = "mid"

    if macd_diff < -0.01:
        macd_state = "neg"
    elif macd_diff > 0.01:
        macd_state = "pos"
    else:
        macd_state = "zero"

    return (rsi_state, macd_state)

def open_position(positions, market, side, entry_price, take_profit, stop_loss, trailing_stop_pct):
    positions[market] = {
        "side": side,
        "entry_price": entry_price,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "trailing_stop_pct": trailing_stop_pct,
        "trailing_stop_price": None
    }

def close_position(positions, market):
    if market in positions:
        del positions[market]

def check_close_conditions(position, current_price):
    side = position["side"]
    tp = position["take_profit"]
    sl = position["stop_loss"]
    trailing_pct = position["trailing_stop_pct"]
    trail_price = position["trailing_stop_price"]
    entry = position["entry_price"]

    if side == "buy":
        if trail_price is None or current_price > trail_price:
            position["trailing_stop_price"] = current_price * (1 - trailing_pct)
            trail_price = position["trailing_stop_price"]
        if current_price >= tp:
            return True, "take_profit"
        if current_price <= sl:
            return True, "stop_loss"
        if trail_price is not None and current_price <= trail_price:
            return True, "trailing_stop"

    elif side == "sell":
        if trail_price is None or current_price < trail_price:
            position["trailing_stop_price"] = current_price * (1 + trailing_pct)
            trail_price = position["trailing_stop_price"]
        if current_price <= tp:
            return True, "take_profit"
        if current_price >= sl:
            return True, "stop_loss"
        if trail_price is not None and current_price >= trail_price:
            return True, "trailing_stop"

    return False, None

def execute_trade_rl(portfolio, positions, market, price):
    base = market.split("-")[0]
    agent = st.session_state["agent"]

    df = get_candles(market, "1h", 50)
    df = add_indicators(df)
    last = df.iloc[-1]
    rsi = last["rsi"]
    macd_diff = last["macd"] - last["macd_signal"]
    state = discretize_state(rsi, macd_diff)

    action = agent.get_action(state)

    log = ""
    pos = positions.get(market, None)

    if pos:
        close, reason = check_close_conditions(pos, price)
        if close:
            side = pos["side"]
            if side == "buy":
                amount_base = portfolio[base]
                if amount_base > 0:
                    portfolio[base] = 0
                    portfolio["EUR"] += amount_base * price
                    log = f"{datetime.now()} - Sluit BUY {amount_base:.6f} {base} @ {price:.2f} ({reason})"
                    close_position(positions, market)
            elif side == "sell":
                log = f"{datetime.now()} - Sluit SELL positie op {market} ({reason})"
                close_position(positions, market)
            return log

    if action == "buy" and portfolio["EUR"] >= 10:
        amount_base = 10 / price
        portfolio["EUR"] -= 10
        portfolio[base] += amount_base

        tp = price * 1.03
        sl = price * 0.98
        trailing = 0.01
        open_position(positions, market, "buy", price, tp, sl, trailing)
        log = f"{datetime.now()} - Koop {amount_base:.6f} {base} @ {price:.2f}, TP: {tp:.2f}, SL: {sl:.2f}, Trailing: {trailing*100}%"

    elif action == "sell" and portfolio[base] >= 0.0001:
        amount_base = portfolio[base]
        portfolio[base] -= amount_base
        portfolio["EUR"] += amount_base * price

        tp = price * 0.97
        sl = price * 1.02
        trailing = 0.01
        open_position(positions, market, "sell", price, tp, sl, trailing)
        log = f"{datetime.now()} - Verkoop {amount_base:.6f} {base} @ {price:.2f}, TP: {tp:.2f}, SL: {sl:.2f}, Trailing: {trailing*100}%"

    else:
        log = f"{datetime.now()} - Houd positie vast op {market}"

    return log

st.title("AI Crypto Trading Platform v0.5 met Q-learning en echte trades")

markets = get_markets()

if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = init_portfolio(markets)
if "positions" not in st.session_state:
    st.session_state["positions"] = init_positions()
if "log" not in st.session_state:
    st.session_state["log"] = []
if "agent" not in st.session_state:
    actions = ["buy", "sell", "hold"]
    states = []
    for rsi_state in ["low", "mid", "high"]:
        for macd_state in ["neg", "zero", "pos"]:
            states.append((rsi_state, macd_state))
    st.session_state["agent"] = QLearningAgent(states, actions)

st.subheader("Portfolio")
for asset, amount in st.session_state["portfolio"].items():
    if asset == "EUR":
        st.write(f"{asset}: €{amount:.2f}")
    else:
        st.write(f"{asset}: {amount:.6f}")

st.subheader("Open Posities")
if st.session_state["positions"]:
    for market, pos in st.session_state["positions"].items():
        st.write(f"{market} - {pos['side'].upper()} - Entry: {pos['entry_price']:.2f} EUR - TP: {pos['take_profit']:.2f} EUR - SL: {pos['stop_loss']:.2f} EUR - Trailing stop: {pos['trailing_stop_pct']*100:.2f}%")
else:
    st.write("Geen open posities")

selected_markets = st.multiselect("Selecteer markten om te traden", markets, default=["BTC-EUR", "ETH-EUR"])

st.subheader("Live Trading met RL Agent")
for market in selected_markets:
    df_live = get_candles(market, interval="1h", limit=50)
    price_live = df_live["close"].iloc[-1]
    st.write(f"Markt: {market} - Prijs: €{price_live:.2f}")

    if st.button(f"Trade RL {market}"):
        result = execute_trade_rl(st.session_state["portfolio"], st.session_state["positions"], market, price_live)
        st.session_state["log"].append(result)
        st.experimental_rerun()

st.subheader("Trading Log (laatste 30 regels)")
for entry in reversed(st.session_state["log"][-30:]):
    st.write(entry)
