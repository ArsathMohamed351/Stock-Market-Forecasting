import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, Flatten


# PAGE CONFIG

st.set_page_config(page_title="📊 MegaStock AI Pro Dashboard", layout="wide")
st.title("📊 MegaStock AI – Pro Dashboard 🚀")


# SIDEBAR CONTROLS

st.sidebar.header("⚙️ Controls")

STOCKS = [
    'AAPL','MSFT','AMZN','GOOG','TSLA','META','NVDA','NFLX','BABA','INTC',
    'ADBE','ORCL','CSCO','IBM','SAP','QCOM','AMD','SHOP','UBER','DIS',
    'PYPL','CRM','V','MA','NKE','KO','PEP'
]

symbol = st.sidebar.selectbox("Stock Symbol", STOCKS)
trade_mode = st.sidebar.radio("Trading Horizon", ["Investment (Daily)", "Intraday (5m / 15m)"])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
future_days = st.sidebar.slider("Future Prediction Days", 5, 180, 30)
force_retrain = st.sidebar.checkbox("Force Model Retrain", False)

intraday_interval = "1d"
if trade_mode == "Intraday (5m / 15m)":
    intraday_interval = st.sidebar.selectbox("Intraday Interval", ["5m", "15m"])

LOOKBACK = 60



# NEW FEATURE: SELECT AI MODEL
selected_model = st.sidebar.selectbox("Select AI Model", ["Weighted Ensemble", "LSTM Only", "Transformer Only"])



# DATA LOADING

@st.cache_data
def load_data(sym, start, end):
    df = yf.download(sym, start=start, end=end, auto_adjust=True)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data
def load_intraday(sym, interval):
    df = yf.download(sym, period="30d", interval=interval, auto_adjust=True, progress=False)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

data = load_data(symbol, start_date, end_date)
idata = data if trade_mode == "Investment (Daily)" else load_intraday(symbol, intraday_interval)

if data.empty or len(data) < LOOKBACK + 20:
    st.error("Not enough data.")
    st.stop()


# LIVE METRICS

latest = float(data['Close'].iloc[-1])
prev = float(data['Close'].iloc[-2])
chg = latest - prev
pct = (chg / prev) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Price", f"${latest:.2f}")
c2.metric("Daily Change", f"${chg:.2f}")
c3.metric("% Change", f"{pct:.2f}%")

# Market Trend
ma50 = data['Close'].rolling(50).mean().iloc[-1]
ma200 = data['Close'].rolling(200).mean().iloc[-1]
if latest > ma50 > ma200:
    trend = "📈 Bullish"
elif latest < ma50 < ma200:
    trend = "📉 Bearish"
else:
    trend = "➖ Sideways"
c4.metric("Market Trend", trend)

# Support & Resistance

support = data['Close'].rolling(20).min().iloc[-1]
resistance = data['Close'].rolling(20).max().iloc[-1]
st.markdown(f"**Support:** ${support:.2f}  \n**Resistance:** ${resistance:.2f}")


# DATA PREPARATION FOR ML

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[['Close']])

X, y = [], []
for i in range(LOOKBACK, len(scaled)):
    X.append(scaled[i-LOOKBACK:i, 0])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


#  MODEL DEFINITIONS

@st.cache_resource
def build_lstm_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK,1)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

@st.cache_resource
def build_transformer_model():
    input_layer = Input(shape=(LOOKBACK, 1))
    x = LayerNormalization()(input_layer)
    x = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer="adam", loss="mse")
    return model


# MODEL TRAINING / LOADING

lstm_model_path = Path(f"{symbol}_lstm.keras")
transformer_model_path = Path(f"{symbol}_transformer.keras")

if lstm_model_path.exists() and not force_retrain:
    lstm_model = load_model(lstm_model_path)
else:
    lstm_model = build_lstm_model()
    lstm_model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
    lstm_model.save(lstm_model_path)

if transformer_model_path.exists() and not force_retrain:
    transformer_model = load_model(transformer_model_path)
else:
    transformer_model = build_transformer_model()
    transformer_model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
    transformer_model.save(transformer_model_path)


# FUTURE PREDICTION

def predict_future(model, scaled_data, days):
    seq = scaled_data[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    future_scaled = []
    for _ in range(days):
        p = model.predict(seq, verbose=0)[0,0]
        future_scaled.append(p)
        seq = np.append(seq[:,1:,:], [[[p]]], axis=1)
    return np.array(future_scaled).reshape(-1,1)

future_lstm = predict_future(lstm_model, scaled, future_days)
future_trans = predict_future(transformer_model, scaled, future_days)

# COMBINE LSTM + TRANSFORMER INTO WEIGHTED FORECAST

pred_test_lstm = lstm_model.predict(X_test, verbose=0)
pred_test_trans = transformer_model.predict(X_test, verbose=0)

rmse_lstm = np.sqrt(mean_squared_error(y_test, pred_test_lstm))
rmse_trans = np.sqrt(mean_squared_error(y_test, pred_test_trans))

# Confidence weights inversely proportional to RMSE
w_lstm = 1 / (rmse_lstm + 1e-6)
w_trans = 1 / (rmse_trans + 1e-6)
w_total = w_lstm + w_trans

future_combined = (future_lstm * w_lstm + future_trans * w_trans) / w_total
future_price_combined = scaler.inverse_transform(future_combined)

# AI FORECAST (TOP OF DASHBOARD)

future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')

st.subheader("🔮 AI Weighted Future Forecast")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(data['Date'], data['Close'], label="Historical")
ax.plot(future_dates, future_price_combined, label="AI Forecast (Weighted)", color='orange')
ax.fill_between(future_dates,
                future_price_combined.flatten() - data['Close'].rolling(20).std().iloc[-1],
                future_price_combined.flatten() + data['Close'].rolling(20).std().iloc[-1],
                alpha=0.1, color='orange')
ax.legend()
st.pyplot(fig)


# RISK METRICS

returns = data['Close'].pct_change().dropna()
sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
cum = (1 + returns).cumprod()
drawdown = cum / cum.cummax() - 1

st.subheader("⚠️ Risk Metrics")
r1, r2 = st.columns(2)
r1.metric("Sharpe Ratio", f"{sharpe:.2f}")
r2.metric("Max Drawdown", f"{drawdown.min()*100:.2f}%")


# TECHNICAL INDICATORS

st.subheader("📊 MACD Indicator")
ema12 = data['Close'].ewm(span=12).mean()
ema26 = data['Close'].ewm(span=26).mean()
macd = ema12 - ema26
signal = macd.ewm(span=9).mean()
fig2, ax2 = plt.subplots(figsize=(12,4))
ax2.plot(data['Date'], macd, label="MACD")
ax2.plot(data['Date'], signal, label="Signal")
ax2.axhline(0, linestyle="--", alpha=0.5)
ax2.legend()
st.pyplot(fig2)

st.subheader("📐 Fibonacci Retracement")
high, low = data['Close'].max(), data['Close'].min()
levels = [0.236, 0.382, 0.5, 0.618]
fig3, ax3 = plt.subplots(figsize=(12,5))
ax3.plot(data['Date'], data['Close'])
for lvl in levels:
    ax3.axhline(high - (high - low) * lvl, linestyle="--")
ax3.axhline(high)
ax3.axhline(low)
st.pyplot(fig3)


# AI DECISION ENGINE (PRO-GRADE)

delta_tf = idata['Close'].diff()
gain_tf = delta_tf.clip(lower=0)
loss_tf = -delta_tf.clip(upper=0)
rs_tf = gain_tf.rolling(14).mean() / loss_tf.rolling(14).mean()
rsi_tf = 100 - (100 / (1 + rs_tf))
ema12_tf, ema26_tf = idata['Close'].ewm(span=12).mean(), idata['Close'].ewm(span=26).mean()
macd_tf, macd_signal_tf = ema12_tf - ema26_tf, (ema12_tf - ema26_tf).ewm(span=9).mean()
macd_hist_tf = macd_tf - macd_signal_tf

tf_ma20 = idata['Close'].rolling(20).mean().iloc[-1]
tf_ma50 = idata['Close'].rolling(50).mean().iloc[-1]
tf_price = float(idata['Close'].iloc[-1])
tf_trend = "Bullish" if tf_price > tf_ma20 > tf_ma50 else "Bearish" if tf_price < tf_ma20 < tf_ma50 else "Sideways"

# AI score

score, reasons = 0, []
if tf_trend == "Bullish": score += 2; reasons.append("Trend aligned bullish")
if 40 < rsi_tf.iloc[-1] < 70: score += 1; reasons.append("RSI healthy")
if macd_hist_tf.iloc[-1] > 0: score += 1; reasons.append("MACD momentum positive")
confidence = max(0, 100 - (rmse_lstm/ latest * 100))
if confidence > 65: score += 1; reasons.append("AI confidence strong")

verdict = "🟢 STRONG BUY" if score >= 5 else "🟡 BUY" if score == 4 else "⏸ WAIT" if score == 3 else "🔴 AVOID" if score == 2 else "❌ STRONG SELL"

st.subheader("🧠 AI Trade Suggestion")
st.markdown(f"""
### **{verdict}**
**Mode:** {trade_mode}  
**AI Confidence:** {confidence:.1f}%

**Reasoning:**  
- {"; ".join(reasons) if reasons else "Market conditions are weak or conflicting"}
""")

# ATR STOP LOSS & TARGET

high_low = idata['High'] - idata['Low']
high_close = np.abs(idata['High'] - idata['Close'].shift())
low_close = np.abs(idata['Low'] - idata['Close'].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
atr = tr.rolling(14).mean().iloc[-1]
sl, tp = tf_price - atr, tf_price + atr*2

st.subheader("📐 AI Risk Levels")
st.write(f"ATR: **{atr:.2f}**")
st.write(f"Suggested Stop-Loss: **${sl:.2f}**")
st.write(f"Suggested Target: **${tp:.2f}**")

# POSITION SIZING

st.subheader("⚖️ Position Sizing")
capital = st.number_input("Total Capital ($)", value=10000)
risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0)
risk_amount = capital * (risk_pct / 100)
position_size = risk_amount / atr if atr > 0 else 0
st.write(f"Risk Amount: **${risk_amount:.2f}**")
st.write(f"Position Size: **{int(position_size)} shares**")


# DATA TABLE
st.subheader("📄 Latest Stock Data")
st.dataframe(data.tail(50))
