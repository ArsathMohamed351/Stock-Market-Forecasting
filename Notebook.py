import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, Flatten

#  PAGE CONFIG

st.set_page_config(page_title="📊 MegaStock AI Pro Dashboard", layout="wide")
st.title("📊 MegaStock AI – Professional Multi-Stock Dashboard 🚀")

#  SIDEBAR CONTROLS

st.sidebar.header("⚙️ Controls")

STOCKS = [
    'AAPL','MSFT','AMZN','GOOG','TSLA','META','NVDA','NFLX','BABA','INTC',
    'ADBE','ORCL','CSCO','IBM','SAP','QCOM','AMD','SHOP','UBER','DIS',
    'PYPL','CRM','V','MA','NKE','KO','PEP'
]

selected_stocks = st.sidebar.multiselect("Select Stocks", STOCKS, default=['AAPL','MSFT','TSLA'])
trade_mode = st.sidebar.radio("Trading Horizon", ["Investment (Daily)", "Intraday (5m / 15m)"])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
future_days = st.sidebar.slider("Future Prediction Days", 5, 180, 30)
force_retrain = st.sidebar.checkbox("Force Model Retrain", False)

intraday_interval = "1d"
if trade_mode == "Intraday (5m / 15m)":
    intraday_interval = st.sidebar.selectbox("Intraday Interval", ["5m", "15m"])

LOOKBACK = 60

# MODEL FUNCTIONS

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

# TOP SECTION: Live Metrics + AI Weighted Forecast + ATR Risk (per stock)

portfolio_summary = []
portfolio_risk_data = {}

for symbol in selected_stocks:
    # Load data
    data = load_data(symbol, start_date, end_date)
    idata = data if trade_mode=="Investment (Daily)" else load_intraday(symbol, intraday_interval)
    if data.empty or len(data) < LOOKBACK + 20:
        st.warning(f"Not enough data for {symbol}")
        continue

    # Scale data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[['Close']])
    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i,0])
        y.append(scaled[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Load/train models
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

# =========================================================
# FUTURE PREDICTION FUNCTION
# =========================================================

def predict_future(model, scaled_data, days):

    # Get last LOOKBACK rows
    seq = scaled_data[-LOOKBACK:]

    # Reshape for deep learning
    seq = seq.reshape(
        1,
        LOOKBACK,
        scaled_data.shape[1]
    )

    future_predictions = []

    for _ in range(days):

        # Predict next close price
        pred = model.predict(
            seq,
            verbose=0
        )[0,0]

        future_predictions.append(pred)

        # Copy last row
        next_row = seq[0, -1].copy()

        # Replace CLOSE column
        # Close index = 3
        next_row[3] = pred

        # Add next timestep
        seq = np.append(
            seq[:,1:,:],
            [[next_row]],
            axis=1
        )

    return np.array(future_predictions)
    
    future_lstm = predict_future(lstm_model, scaled, future_days)
    future_trans = predict_future(transformer_model, scaled, future_days)

    # Weighted AI forecast
    rmse_lstm = np.sqrt(mean_squared_error(y_test, lstm_model.predict(X_test, verbose=0)))
    rmse_trans = np.sqrt(mean_squared_error(y_test, transformer_model.predict(X_test, verbose=0)))
    w_lstm = 1/(rmse_lstm+1e-6)
    w_trans = 1/(rmse_trans+1e-6)
    future_combined = (future_lstm*w_lstm + future_trans*w_trans)/(w_lstm+w_trans)
    future_price_combined = scaler.inverse_transform(future_combined)

    # Latest metrics
    latest = float(data['Close'].iloc[-1])
    prev = float(data['Close'].iloc[-2])
    chg = latest - prev
    pct = (chg/prev)*100

    # Trend
    ma50 = data['Close'].rolling(50).mean().iloc[-1]
    ma200 = data['Close'].rolling(200).mean().iloc[-1]
    trend = "Bullish" if latest>ma50>ma200 else "Bearish" if latest<ma50<ma200 else "Sideways"

    # ATR Risk
    high_low = idata['High'] - idata['Low']
    high_close = np.abs(idata['High'] - idata['Close'].shift())
    low_close = np.abs(idata['Low'] - idata['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    sl = idata['Close'] - atr
    tp = idata['Close'] + atr*2
    portfolio_risk_data[symbol] = {'Date': idata['Date'], 'Close': idata['Close'], 'SL': sl, 'TP': tp}

    # AI Score
    delta = idata['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean()/loss.rolling(14).mean()
    rsi = 100 - (100/(1+rs))
    ema12, ema26 = idata['Close'].ewm(span=12).mean(), idata['Close'].ewm(span=26).mean()
    macd, macd_signal = ema12-ema26, (ema12-ema26).ewm(span=9).mean()
    macd_hist = macd - macd_signal
    score, reasons = 0, []
    if trend=="Bullish": score+=2; reasons.append("Trend bullish")
    if 40<rsi.iloc[-1]<70: score+=1; reasons.append("RSI healthy")
    if macd_hist.iloc[-1]>0: score+=1; reasons.append("MACD positive")
    confidence = max(0, 100-(rmse_lstm/latest*100))
    if confidence>65: score+=1; reasons.append("AI confidence strong")
    verdict = "STRONG BUY" if score>=5 else "BUY" if score==4 else "WAIT" if score==3 else "AVOID" if score==2 else "STRONG SELL"

    portfolio_summary.append({
        "Stock": symbol,
        "Latest": latest,
        "Change": chg,
        "%Change": pct,
        "Trend": trend,
        "RSI": round(rsi.iloc[-1],2),
        "MACD": "Positive" if macd_hist.iloc[-1]>0 else "Negative",
        "ATR": round(atr.iloc[-1],2),
        "Confidence": round(confidence,1),
        "Verdict": verdict
    })

    # AI Forecast Graph

    st.subheader(f"🔮 {symbol} AI Weighted Forecast")
    future_dates = pd.date_range(start=data['Date'].iloc[-1]+pd.Timedelta(days=1), periods=future_days, freq='B')
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(data['Date'], data['Close'], label="Historical")
    ax.plot(future_dates, future_price_combined, label="AI Forecast", color='orange')
    ax.fill_between(future_dates,
                    future_price_combined.flatten()-data['Close'].rolling(20).std().iloc[-1],
                    future_price_combined.flatten()+data['Close'].rolling(20).std().iloc[-1],
                    alpha=0.1, color='orange')
    ax.legend()
    st.pyplot(fig)

    # ATR Risk Graph (per stock)
    
    st.subheader(f"📊 {symbol} ATR Risk Management")
    fig2, ax2 = plt.subplots(figsize=(12,4))
    ax2.plot(idata['Date'], idata['Close'], label="Price", color='blue')
    ax2.plot(idata['Date'], sl, linestyle='--', label="Stop-Loss", color='red')
    ax2.plot(idata['Date'], tp, linestyle='--', label="Target", color='green')
    ax2.fill_between(idata['Date'], sl, tp, color='orange', alpha=0.1)
    ax2.legend()
    st.pyplot(fig2)

# MIDDLE SECTION: Portfolio Summary & Risk

st.subheader("📊 Multi-Stock AI Portfolio Summary")
st.dataframe(pd.DataFrame(portfolio_summary), use_container_width=True)

st.subheader("📉 Portfolio Drawdown Heatmap")
drawdown_matrix = []
for symbol in selected_stocks:
    data = load_data(symbol, start_date, end_date)
    returns = data['Close'].pct_change().dropna()
    cum = (1+returns).cumprod()
    dd = cum/cum.cummax()-1
    drawdown_matrix.append(dd.values[-len(dd):])
drawdown_df = pd.DataFrame(drawdown_matrix, index=selected_stocks).T
fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(drawdown_df, cmap="Reds", cbar_kws={'label':'Drawdown'}, ax=ax)
st.pyplot(fig)

st.subheader("📊 Portfolio ATR Risk Management Graph (All Stocks)")
fig, ax = plt.subplots(figsize=(14,6))
for symbol, df in portfolio_risk_data.items():
    ax.plot(df['Date'], df['Close'], label=f"{symbol} Price")
    ax.plot(df['Date'], df['SL'], linestyle='--', label=f"{symbol} Stop-Loss", alpha=0.7)
    ax.plot(df['Date'], df['TP'], linestyle='--', label=f"{symbol} Target", alpha=0.7)
    ax.fill_between(df['Date'], df['SL'], df['TP'], alpha=0.1)
ax.set_title("Portfolio ATR Risk Levels")
ax.set_ylabel("Price ($)")
ax.legend()
st.pyplot(fig)

# BOTTOM SECTION: Technical Indicators (optional)

st.subheader("📊 Technical Indicators (Reference / Optional)")
for symbol in selected_stocks:
    data = load_data(symbol, start_date, end_date)
    st.markdown(f"### {symbol} MACD")
    ema12 = data['Close'].ewm(span=12).mean()
    ema26 = data['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(data['Date'], macd, label="MACD")
    ax.plot(data['Date'], signal, label="Signal")
    ax.axhline(0, linestyle="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"### {symbol} Fibonacci")
    high, low = data['Close'].max(), data['Close'].min()
    levels = [0.236, 0.382, 0.5, 0.618]
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(data['Date'], data['Close'])
    for lvl in levels:
        ax.axhline(high - (high - low) * lvl, linestyle="--")
    ax.axhline(high)
    ax.axhline(low)
    st.pyplot(fig)

st.success("✅ Dashboard Ready: AI & Risk graphs on top, Technical Indicators at bottom.")
