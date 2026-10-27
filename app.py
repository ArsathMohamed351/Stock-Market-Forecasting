# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, Flatten


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Market Lens",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── PREMIUM CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Base ── */
.stApp {
    background: #08090d;
    color: #e8eaf0;
}
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0c0d11 !important;
    border-right: 1px solid #1c1e26 !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #4a5568 !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] input {
    background: #111318 !important;
    border: 1px solid #1c1e26 !important;
    border-radius: 4px !important;
    color: #e8eaf0 !important;
}
[data-testid="stSidebar"] hr { border-color: #1c1e26 !important; }

/* ── Hero ── */
.hero {
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid #1c1e26;
    margin-bottom: 2.5rem;
}
.hero-eyebrow {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00d4ff;
    margin-bottom: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1;
    margin: 0;
}
.hero-ticker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #2d3748;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
}

/* ── Stat blocks ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 0;
    border-top: 1px solid #1c1e26;
    margin-bottom: 2.5rem;
}
.stat-block {
    padding: 1.4rem 1rem 1.2rem;
    border-right: 1px solid #1c1e26;
    border-bottom: 1px solid #1c1e26;
}
.stat-block:last-child { border-right: none; }
.stat-label {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #2d3748;
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}
.stat-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}
.stat-delta {
    font-size: 0.72rem;
    font-weight: 500;
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}
.up   { color: #00e5a0; }
.down { color: #ff4757; }
.flat { color: #4a5568; }

/* ── Section label ── */
.sec-label {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #2d3748;
    font-family: 'JetBrains Mono', monospace;
    margin: 2.5rem 0 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #1c1e26;
}

/* ── AI Decision ── */
.ai-panel {
    display: grid;
    grid-template-columns: 220px 1fr;
    gap: 0;
    border: 1px solid #1c1e26;
    margin: 1rem 0;
}
.ai-verdict-block {
    padding: 2rem 1.5rem;
    border-right: 1px solid #1c1e26;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.ai-verdict-word {
    font-size: 1.5rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    line-height: 1;
}
.ai-verdict-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #2d3748;
    margin-top: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.ai-score-bar {
    height: 2px;
    background: #1c1e26;
    margin-top: 0.5rem;
    position: relative;
}
.ai-score-fill {
    height: 100%;
    position: absolute;
    top: 0; left: 0;
    transition: width 0.4s;
}
.ai-reasons-block {
    padding: 2rem;
}
.ai-reasons-title {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #2d3748;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 1rem;
}
.ai-reason {
    font-size: 0.82rem;
    color: #4a5568;
    padding: 0.3rem 0;
    border-bottom: 1px solid #111318;
    font-family: 'JetBrains Mono', monospace;
}
.ai-reason.active { color: #e8eaf0; }
.ai-chips {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.ai-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.ai-chip-label { color: #2d3748; }
.ai-chip-val   { color: #e8eaf0; font-weight: 600; }

/* ── Streamlit metric override ── */
[data-testid="stMetric"] {
    background: #0c0d11 !important;
    border: 1px solid #1c1e26 !important;
    border-radius: 0 !important;
    padding: 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    color: #2d3748 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
}

/* ── Streamlit widget overrides ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1c1e26 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #2d3748 !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    font-family: 'JetBrains Mono', monospace !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 0 !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* Spinner & data frame */
[data-testid="stDataFrame"] { border-radius: 0 !important; }
.stDataFrame thead th {
    background: #111318 !important;
    font-size: 0.65rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #2d3748 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #08090d; }
::-webkit-scrollbar-thumb { background: #1c1e26; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root { --bg: #0b0d10; --panel: #12161c; --line: #252b33; --text: #f1f4f8; --muted: #8e99a8; --accent: #8ba7ff; }
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stAppViewContainer"] > .main { background: var(--bg); }
.block-container { max-width: 1480px; padding: 2rem 2.5rem 3.5rem; }
[data-testid="stSidebar"] { background: #0e1116 !important; border-right: 1px solid var(--line) !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
[data-testid="stSidebar"] .stSelectbox > div > div, [data-testid="stSidebar"] [data-baseweb="input"] { background: #151a21 !important; border-color: #2b333d !important; border-radius: 6px !important; }
.hero { padding: .4rem 0 1.75rem; margin-bottom: 1.5rem; border-bottom: 1px solid var(--line); }
.hero-eyebrow { color: var(--accent); font-size: .68rem; letter-spacing: .16em; margin-bottom: .45rem; }
.hero-title { font-size: 2.25rem; font-weight: 700; letter-spacing: -.045em; }
.hero-ticker { color: var(--muted); font-size: .78rem; margin-top: .65rem; }
.stat-row { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; margin: 0 0 2.25rem; }
.stat-block { background: var(--panel); padding: 1.15rem 1.2rem; border-bottom: 0; border-right: 1px solid var(--line); }
.stat-label, [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .65rem !important; letter-spacing: .1em !important; }
.stat-value, [data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.4rem !important; font-weight: 650 !important; }
.stat-delta { color: var(--muted); }
.up { color: #74c69d; } .down { color: #ef8f8f; } .flat { color: var(--muted); }
.section-header { color: var(--text); font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; padding: 1.75rem 0 .75rem; margin: 0; border-bottom: 1px solid var(--line); }
[data-testid="stMetric"] { background: var(--panel) !important; border: 1px solid var(--line) !important; border-radius: 8px !important; padding: 1rem 1.15rem !important; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; padding: .7rem 1rem !important; }
.stTabs [aria-selected="true"] { color: var(--text) !important; border-bottom-color: var(--accent) !important; }
.verdict-card, .signal-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 1.35rem; min-height: 176px; }
.card-label { color: var(--muted); font: 600 .65rem 'JetBrains Mono', monospace; letter-spacing: .12em; text-transform: uppercase; }
.verdict-value { margin-top: .8rem; color: var(--text); font-size: 1.6rem; font-weight: 700; letter-spacing: -.03em; }
.verdict-score { margin-top: .7rem; color: var(--muted); font-size: .75rem; }
.score-track { background: #252b33; height: 4px; border-radius: 99px; margin-top: .65rem; overflow: hidden; }
.score-fill { background: var(--accent); height: 100%; border-radius: 99px; }
.signal-row { padding: .45rem 0; color: #c6ced9; font-size: .85rem; border-bottom: 1px solid #20262e; }
.signal-row:last-child { border: 0; }
.signal-summary { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-top: 1rem; color: var(--muted); font: .72rem 'JetBrains Mono', monospace; }
.signal-summary strong { color: var(--text); font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

LOOKBACK   = 60
CHART_BG   = "rgba(8,9,13,0)"
GRID_COLOR = "rgba(28,30,38,1)"
ACCENT     = "#00d4ff"
ACCENT2    = "#7f8cff"
GREEN      = "#00e5a0"
RED        = "#ff4757"
ORANGE     = "#ffaa00"

STOCKS = [
    'AAPL','MSFT','AMZN','GOOG','TSLA','META','NVDA','NFLX','BABA','INTC',
    'ADBE','ORCL','CSCO','IBM','SAP','QCOM','AMD','SHOP','UBER','DIS',
    'PYPL','CRM','V','MA','NKE','KO','PEP'
]


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding: 1.5rem 0 1rem;'>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:#00d4ff;margin-bottom:0.4rem;'>Market Lens</div>
        <div style='font-size:1.4rem;font-weight:900;color:#fff;letter-spacing:-0.03em;'>AI Terminal</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    symbol        = st.selectbox("Stock Symbol", STOCKS)
    trade_mode    = st.radio("Trading Horizon", ["Investment (Daily)", "Intraday (5m / 15m)"])
    start_date    = st.date_input("Start Date", pd.to_datetime("2021-01-01"))
    end_date      = st.date_input("End Date",   pd.to_datetime("today"))
    future_days   = st.slider("Forecast Horizon (days)", 5, 180, 30)
    st.markdown("---")
    selected_model = st.selectbox("AI Model", ["Weighted Ensemble", "LSTM Only", "Transformer Only"])
    force_retrain  = st.checkbox("Force Retrain", False)
    intraday_interval = "1d"
    if trade_mode == "Intraday (5m / 15m)":
        intraday_interval = st.selectbox("Interval", ["5m", "15m"])
    st.markdown("---")
    st.markdown("<span style='font-family:JetBrains Mono,monospace;font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;color:#2d3748;'>Position Sizing</span>", unsafe_allow_html=True)
    capital  = st.number_input("Capital ($)", value=10000, step=500)
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, step=0.5)


# ── HERO ──────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">Market Intelligence Terminal</div>
    <div class="hero-title">Market Lens</div>
    <div class="hero-ticker">{symbol} &nbsp;|&nbsp; {trade_mode} &nbsp;|&nbsp; LSTM + Transformer Ensemble</div>
</div>
""", unsafe_allow_html=True)


# ── DATA LOADING ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(sym, start, end):
    df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Datetime" in df.columns and "Date" not in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)
    return df

@st.cache_data(show_spinner=False)
def load_intraday(sym, interval):
    df = yf.download(sym, period="30d", interval=interval, auto_adjust=True, progress=False)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Datetime" in df.columns and "Date" not in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)
    return df

if start_date >= end_date:
    st.error("Start Date must be earlier than End Date.")
    st.stop()

with st.spinner("Fetching market data…"):
    data  = load_data(symbol, start_date, end_date)
    idata = data if trade_mode == "Investment (Daily)" else load_intraday(symbol, intraday_interval)

if data.empty or len(data) < LOOKBACK + 20 or not {"Close", "Date"}.issubset(data.columns):
    st.error("Not enough historical data. Try a wider date range.")
    st.stop()


# ── LIVE METRICS ─────────────────────────────────────────────────────────────

if idata.empty or len(idata) < 20 or not {"High", "Low", "Close", "Date"}.issubset(idata.columns):
    st.error("Not enough intraday data to calculate indicators. Choose another interval or try again later.")
    st.stop()

latest = float(data['Close'].iloc[-1])
prev   = float(data['Close'].iloc[-2])
chg    = latest - prev
pct    = (chg / prev) * 100

ma50  = float(data['Close'].rolling(50).mean().iloc[-1])
ma200 = float(data['Close'].rolling(200).mean().iloc[-1])
if latest > ma50 > ma200:   trend = "Bullish"
elif latest < ma50 < ma200: trend = "Bearish"
else:                       trend = "Sideways"

support    = float(data['Close'].rolling(20).min().iloc[-1])
resistance = float(data['Close'].rolling(20).max().iloc[-1])
returns    = data['Close'].pct_change().dropna()
sharpe     = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0.0
cum        = (1 + returns).cumprod()
max_dd     = float((cum / cum.cummax() - 1).min() * 100)

# compute reusable values
delta_cls  = "up" if chg >= 0 else "down"
delta_sign = "+" if chg >= 0 else ""
trend_raw  = "Bullish" if latest > ma50 > ma200 else "Bearish" if latest < ma50 < ma200 else "Sideways"
trend_cls  = "up" if trend_raw == "Bullish" else "down" if trend_raw == "Bearish" else "flat"

st.markdown(f"""
<div class="stat-row">
    <div class="stat-block">
        <div class="stat-label">Last Price</div>
        <div class="stat-value">${latest:,.2f}</div>
        <div class="stat-delta {delta_cls}">{delta_sign}${chg:.2f}</div>
    </div>
    <div class="stat-block">
        <div class="stat-label">Daily Change</div>
        <div class="stat-value {delta_cls}">{delta_sign}{pct:.2f}%</div>
        <div class="stat-delta flat">vs yesterday</div>
    </div>
    <div class="stat-block">
        <div class="stat-label">Trend</div>
        <div class="stat-value {trend_cls}" style="font-size:1.3rem;">{trend_raw}</div>
        <div class="stat-delta flat">MA 50 / MA 200</div>
    </div>
    <div class="stat-block">
        <div class="stat-label">Sharpe Ratio</div>
        <div class="stat-value">{sharpe:.2f}</div>
        <div class="stat-delta flat">annualised</div>
    </div>
    <div class="stat-block">
        <div class="stat-label">Max Drawdown</div>
        <div class="stat-value down">{max_dd:.1f}%</div>
        <div class="stat-delta flat">historical</div>
    </div>
    <div class="stat-block">
        <div class="stat-label">Support / Resistance</div>
        <div class="stat-value" style="font-size:1.1rem;">${support:.0f} / ${resistance:.0f}</div>
        <div class="stat-delta flat">20-day range</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── CANDLESTICK CHART ────────────────────────────────────────────────────────

st.markdown('<p class="section-header">Price History & Moving Averages</p>', unsafe_allow_html=True)

data['MA20']  = data['Close'].rolling(20).mean()
data['MA50']  = data['Close'].rolling(50).mean()
data['MA200'] = data['Close'].rolling(200).mean()

fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          row_heights=[0.75, 0.25], vertical_spacing=0.04)

if all(c in data.columns for c in ['Open', 'High', 'Low']):
    fig_price.add_trace(go.Candlestick(
        x=data['Date'], open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'], name="OHLC",
        increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor=GREEN,  decreasing_fillcolor=RED,
    ), row=1, col=1)
else:
    fig_price.add_trace(go.Scatter(
        x=data['Date'], y=data['Close'], name="Close",
        line=dict(color=ACCENT, width=1.5)
    ), row=1, col=1)

for col_name, color, name in [('MA20', ORANGE, 'MA 20'), ('MA50', ACCENT, 'MA 50'), ('MA200', ACCENT2, 'MA 200')]:
    fig_price.add_trace(go.Scatter(
        x=data['Date'], y=data[col_name], name=name,
        line=dict(color=color, width=1.2, dash='dot'), opacity=0.8
    ), row=1, col=1)

if 'Volume' in data.columns:
    vol_colors = [GREEN if i == 0 or data['Close'].iloc[i] >= data['Close'].iloc[i-1] else RED
                  for i in range(len(data))]
    fig_price.add_trace(go.Bar(
        x=data['Date'], y=data['Volume'], name="Volume",
        marker_color=vol_colors, opacity=0.55
    ), row=2, col=1)

fig_price.update_layout(
    template="plotly_dark", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
    height=520, margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    xaxis_rangeslider_visible=False,
    yaxis=dict(gridcolor=GRID_COLOR, tickprefix="$"),
    yaxis2=dict(gridcolor=GRID_COLOR),
    xaxis2=dict(showgrid=False),
)
st.plotly_chart(fig_price, use_container_width=True)


# ── ML DATA PREP ─────────────────────────────────────────────────────────────

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[['Close']])
X, y = [], []
for i in range(LOOKBACK, len(scaled)):
    X.append(scaled[i - LOOKBACK:i, 0])
    y.append(scaled[i, 0])
X, y  = np.array(X), np.array(y)
X     = X.reshape(X.shape[0], X.shape[1], 1)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# ── MODEL DEFINITIONS ────────────────────────────────────────────────────────

def build_lstm_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.3), LSTM(64), Dropout(0.3), Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_transformer_model():
    inp = Input(shape=(LOOKBACK, 1))
    x   = LayerNormalization()(inp)
    x   = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x   = Flatten()(x)
    x   = Dense(64, activation='relu')(x)
    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model


# ── MODEL TRAINING / LOADING ─────────────────────────────────────────────────

lstm_path  = Path(f"{symbol}_lstm.keras")
trans_path = Path(f"{symbol}_transformer.keras")

with st.spinner("Loading AI models…"):
    if lstm_path.exists() and not force_retrain:
        lstm_model = load_model(lstm_path)
    else:
        with st.spinner(f"Training LSTM for {symbol}…"):
            lstm_model = build_lstm_model()
            lstm_model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
            lstm_model.save(lstm_path)

    if trans_path.exists() and not force_retrain:
        transformer_model = load_model(trans_path)
    else:
        with st.spinner(f"Training Transformer for {symbol}…"):
            transformer_model = build_transformer_model()
            transformer_model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
            transformer_model.save(trans_path)


# ── PREDICTIONS ───────────────────────────────────────────────────────────────

def predict_future(model, scaled_data, days):
    seq = scaled_data[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    preds = []
    for _ in range(days):
        p = model.predict(seq, verbose=0)[0, 0]
        preds.append(p)
        seq = np.append(seq[:, 1:, :], [[[p]]], axis=1)
    return np.array(preds).reshape(-1, 1)

future_lstm  = predict_future(lstm_model, scaled, future_days)
future_trans = predict_future(transformer_model, scaled, future_days)

pred_lstm  = lstm_model.predict(X_test, verbose=0)
pred_trans = transformer_model.predict(X_test, verbose=0)
rmse_lstm  = np.sqrt(mean_squared_error(y_test, pred_lstm))
rmse_trans = np.sqrt(mean_squared_error(y_test, pred_trans))
w_lstm, w_trans = 1 / (rmse_lstm + 1e-6), 1 / (rmse_trans + 1e-6)

future_combined    = (future_lstm * w_lstm + future_trans * w_trans) / (w_lstm + w_trans)
future_price_comb  = scaler.inverse_transform(future_combined).flatten()
future_price_lstm  = scaler.inverse_transform(future_lstm).flatten()
future_price_trans = scaler.inverse_transform(future_trans).flatten()

future_dates = pd.date_range(
    start=data['Date'].iloc[-1] + pd.Timedelta(days=1),
    periods=future_days, freq='B'
)
confidence = max(0.0, 100.0 - (rmse_lstm / (latest + 1e-6)) * 100)
std_20     = float(data['Close'].rolling(20).std().iloc[-1])


# ── AI FORECAST CHART ────────────────────────────────────────────────────────

st.markdown('<p class="section-header">AI Price Forecast</p>', unsafe_allow_html=True)

hist_slice = data[data['Date'] >= data['Date'].iloc[-1] - pd.Timedelta(days=180)]

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(
    x=hist_slice['Date'], y=hist_slice['Close'],
    name="Historical", line=dict(color=ACCENT, width=2)
))

if selected_model in ("Weighted Ensemble", "LSTM Only"):
    fp   = future_price_comb if selected_model == "Weighted Ensemble" else future_price_lstm
    name = "Weighted Ensemble" if selected_model == "Weighted Ensemble" else "LSTM Forecast"
    fig_fc.add_trace(go.Scatter(x=future_dates, y=fp, name=name,
                                line=dict(color=ORANGE, width=2.5)))
    fig_fc.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(fp + std_20) + list((fp - std_20)[::-1]),
        fill='toself', fillcolor='rgba(237,137,54,0.1)',
        line=dict(color='rgba(0,0,0,0)'), name="Confidence band"
    ))
else:
    fig_fc.add_trace(go.Scatter(x=future_dates, y=future_price_trans,
                                name="Transformer Forecast",
                                line=dict(color=ACCENT2, width=2.5)))
    fig_fc.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(future_price_trans + std_20) + list((future_price_trans - std_20)[::-1]),
        fill='toself', fillcolor='rgba(159,122,234,0.1)',
        line=dict(color='rgba(0,0,0,0)'), name="Confidence band"
    ))

fig_fc.add_vline(x=str(data['Date'].iloc[-1]), line_width=1,
                 line_dash="dash", line_color="rgba(255,255,255,0.2)")
fig_fc.add_annotation(x=str(data['Date'].iloc[-1]), y=1, yref="paper",
                      text="Today", showarrow=False,
                      font=dict(color="#718096", size=11), xshift=6, yshift=-14)

fig_fc.update_layout(
    template="plotly_dark", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
    height=420, margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    yaxis=dict(gridcolor=GRID_COLOR, tickprefix="$"),
    xaxis=dict(gridcolor=GRID_COLOR), hovermode="x unified"
)
st.plotly_chart(fig_fc, use_container_width=True)

ac1, ac2, ac3 = st.columns(3)
ac1.metric("LSTM RMSE",        f"{rmse_lstm:.5f}")
ac2.metric("Transformer RMSE", f"{rmse_trans:.5f}")
ac3.metric("AI Confidence",    f"{confidence:.1f}%")


# ── AI TRADE DECISION ────────────────────────────────────────────────────────

st.markdown('<p class="section-header">AI Trade Decision</p>', unsafe_allow_html=True)

delta_tf    = idata['Close'].diff()
gain_tf     = delta_tf.clip(lower=0)
loss_tf     = -delta_tf.clip(upper=0)
rs_tf       = gain_tf.rolling(14).mean() / loss_tf.rolling(14).mean()
rsi_tf      = 100 - (100 / (1 + rs_tf))
ema12_tf    = idata['Close'].ewm(span=12).mean()
ema26_tf    = idata['Close'].ewm(span=26).mean()
macd_tf     = ema12_tf - ema26_tf
macd_sig_tf = macd_tf.ewm(span=9).mean()
macd_hist   = macd_tf - macd_sig_tf

tf_ma20  = float(idata['Close'].rolling(20).mean().iloc[-1])
tf_ma50  = float(idata['Close'].rolling(50).mean().iloc[-1])
tf_price = float(idata['Close'].iloc[-1])
rsi_val  = float(rsi_tf.iloc[-1])
tf_trend = "Bullish" if tf_price > tf_ma20 > tf_ma50 else \
           "Bearish" if tf_price < tf_ma20 < tf_ma50 else "Sideways"

score, reasons = 0, []
if tf_trend == "Bullish":   score += 2; reasons.append("Trend is aligned to the upside")
if 40 < rsi_val < 70:       score += 1; reasons.append(f"RSI is balanced ({rsi_val:.1f})")
if macd_hist.iloc[-1] > 0:  score += 1; reasons.append("MACD momentum is positive")
if confidence > 65:         score += 1; reasons.append(f"Model confidence is strong ({confidence:.1f}%)")

verdict_map = {
    5: ("Strong Buy",  "verdict-strong-buy"),
    4: ("Buy",         "verdict-buy"),
    3: ("Wait",        "verdict-wait"),
    2: ("Avoid",       "verdict-avoid"),
    1: ("Strong Sell", "verdict-strong-sell"),
    0: ("Strong Sell", "verdict-strong-sell"),
}
verdict_text, verdict_cls = verdict_map.get(score, ("Wait", "verdict-wait"))
reasons_html = "".join([f"<div class='signal-row'>{r}</div>" for r in reasons]) \
               if reasons else "<div class='signal-row'>Market conditions are mixed.</div>"

dec1, dec2 = st.columns([1, 2])
with dec1:
    st.markdown(f"""
    <div class="verdict-card">
        <div class="card-label">Model signal</div>
        <div class="verdict-value {verdict_cls}">{verdict_text}</div>
        <div class="verdict-score">Composite score: {score} / 5</div>
        <div class="score-track">
            <div class="score-fill" style="width:{score/5*100:.0f}%;"></div>
        </div>
        <div class="verdict-score">{trade_mode}</div>
    </div>""", unsafe_allow_html=True)

with dec2:
    st.markdown(f"""
    <div class="signal-card">
        <div class="card-label" style="margin-bottom:0.7rem;">Signal drivers</div>
        {reasons_html}
        <div class="signal-summary">
            <span>RSI <strong>{rsi_val:.1f}</strong></span>
            <span>Trend <strong>{tf_trend}</strong></span>
            <span>MACD <strong>{'Positive' if macd_hist.iloc[-1] > 0 else 'Negative'}</strong></span>
        </div>
    </div>""", unsafe_allow_html=True)


# ── ATR RISK MANAGEMENT ───────────────────────────────────────────────────────

st.markdown('<p class="section-header">Risk Management</p>', unsafe_allow_html=True)

high_low   = idata['High'] - idata['Low']
high_close = np.abs(idata['High'] - idata['Close'].shift())
low_close  = np.abs(idata['Low']  - idata['Close'].shift())
tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
atr        = float(tr.rolling(14).mean().iloc[-1])
sl         = tf_price - atr
tp         = tf_price + atr * 2
risk_amount   = capital * (risk_pct / 100)
position_size = int(risk_amount / atr) if atr > 0 else 0

r1, r2, r3, r4 = st.columns(4)
r1.metric("ATR (14)",      f"${atr:.2f}")
r2.metric("Stop Loss",     f"${sl:.2f}", delta=f"-${atr:.2f}", delta_color="inverse")
r3.metric("Price Target",  f"${tp:.2f}", delta=f"+${atr*2:.2f}")
r4.metric("Position Size", f"{position_size} shares", delta=f"${risk_amount:.0f} at risk")

date_col  = 'Datetime' if 'Datetime' in idata.columns else 'Date'
atr_trail = tr.rolling(14).mean()
fig_atr   = go.Figure()
fig_atr.add_trace(go.Scatter(x=idata[date_col], y=idata['Close'],
                             name="Price", line=dict(color=ACCENT, width=2)))
fig_atr.add_trace(go.Scatter(x=idata[date_col], y=idata['Close'] - atr_trail,
                             name="Stop-Loss (ATR)", line=dict(color=RED, width=1.5, dash='dash')))
fig_atr.add_trace(go.Scatter(x=idata[date_col], y=idata['Close'] + atr_trail * 2,
                             name="Target (2x ATR)", line=dict(color=GREEN, width=1.5, dash='dash'),
                             fill='tonexty', fillcolor='rgba(72,187,120,0.04)'))
fig_atr.update_layout(
    template="plotly_dark", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
    height=340, margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    yaxis=dict(gridcolor=GRID_COLOR, tickprefix="$"),
    xaxis=dict(gridcolor=GRID_COLOR), hovermode="x unified"
)
st.plotly_chart(fig_atr, use_container_width=True)


# ── TECHNICAL INDICATORS ─────────────────────────────────────────────────────

st.markdown('<p class="section-header">Technical Indicators</p>', unsafe_allow_html=True)

tab_macd, tab_rsi, tab_fib = st.tabs(["MACD", "RSI", "Fibonacci"])

with tab_macd:
    ema12_d = data['Close'].ewm(span=12).mean()
    ema26_d = data['Close'].ewm(span=26).mean()
    macd_d  = ema12_d - ema26_d
    sig_d   = macd_d.ewm(span=9).mean()
    hist_d  = macd_d - sig_d
    hcolors = [GREEN if v >= 0 else RED for v in hist_d]
    fig_m   = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.55, 0.45], vertical_spacing=0.05)
    fig_m.add_trace(go.Scatter(x=data['Date'], y=data['Close'],
                               name="Close", line=dict(color=ACCENT, width=1.5)), row=1, col=1)
    fig_m.add_trace(go.Scatter(x=data['Date'], y=macd_d,
                               name="MACD", line=dict(color=ACCENT, width=1.5)), row=2, col=1)
    fig_m.add_trace(go.Scatter(x=data['Date'], y=sig_d,
                               name="Signal", line=dict(color=ORANGE, width=1.5)), row=2, col=1)
    fig_m.add_trace(go.Bar(x=data['Date'], y=hist_d, name="Histogram",
                           marker_color=hcolors, opacity=0.7), row=2, col=1)
    fig_m.update_layout(template="plotly_dark", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                        height=400, margin=dict(l=0, r=0, t=10, b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                        yaxis=dict(gridcolor=GRID_COLOR, tickprefix="$"), yaxis2=dict(gridcolor=GRID_COLOR),
                        hovermode="x unified")
    st.plotly_chart(fig_m, use_container_width=True)

with tab_rsi:
    delta_r = data['Close'].diff()
    gain_r  = delta_r.clip(lower=0)
    loss_r  = -delta_r.clip(upper=0)
    rs_r    = gain_r.rolling(14).mean() / loss_r.rolling(14).mean()
    rsi_r   = 100 - (100 / (1 + rs_r))
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data['Date'], y=rsi_r, name="RSI (14)",
                                 line=dict(color=ACCENT2, width=2),
                                 fill='tozeroy', fillcolor='rgba(159,122,234,0.06)'))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color=RED,
                      annotation_text="Overbought 70", annotation_position="right")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color=GREEN,
                      annotation_text="Oversold 30", annotation_position="right")
    fig_rsi.add_hrect(y0=30, y1=70, fillcolor="rgba(99,179,237,0.03)", line_width=0)
    fig_rsi.update_layout(template="plotly_dark", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                          height=340, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis=dict(gridcolor=GRID_COLOR, range=[0, 100]),
                          xaxis=dict(gridcolor=GRID_COLOR), hovermode="x unified")
    st.plotly_chart(fig_rsi, use_container_width=True)

with tab_fib:
    high_p = float(data['Close'].max())
    low_p  = float(data['Close'].min())
    diff_p = high_p - low_p
    fib_levels = {
        "0.0% (High)": high_p, "23.6%": high_p - diff_p * 0.236,
        "38.2%": high_p - diff_p * 0.382, "50.0%": high_p - diff_p * 0.500,
        "61.8%": high_p - diff_p * 0.618, "100% (Low)": low_p,
    }
    fib_colors = [ACCENT2, ACCENT, GREEN, ORANGE, RED, "#718096"]
    fig_fib = go.Figure()
    fig_fib.add_trace(go.Scatter(x=data['Date'], y=data['Close'],
                                 name="Close", line=dict(color=ACCENT, width=1.5)))
    for (lvl_name, lvl_val), c in zip(fib_levels.items(), fib_colors):
        fig_fib.add_hline(y=lvl_val, line_dash="dot", line_color=c,
                          annotation_text=f"{lvl_name}: ${lvl_val:,.0f}",
                          annotation_position="right", annotation_font_size=11)
    fig_fib.update_layout(template="plotly_dark", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                          height=380, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis=dict(gridcolor=GRID_COLOR, tickprefix="$"),
                          xaxis=dict(gridcolor=GRID_COLOR), hovermode="x unified")
    st.plotly_chart(fig_fib, use_container_width=True)


# ── RETURNS ANALYSIS ─────────────────────────────────────────────────────────

st.markdown('<p class="section-header">Returns & Risk Analysis</p>', unsafe_allow_html=True)

ra1, ra2 = st.columns(2)
with ra1:
    fig_ret = go.Figure()
    fig_ret.add_trace(go.Histogram(x=returns * 100, nbinsx=60, name="Daily Returns",
                                   marker=dict(color=ACCENT, opacity=0.7,
                                               line=dict(width=0.5, color=CHART_BG))))
    fig_ret.add_vline(x=float(returns.mean() * 100), line_dash="dash", line_color=GREEN,
                      annotation_text=f"Mean {returns.mean()*100:.3f}%")
    fig_ret.update_layout(
        title=dict(text="Daily Return Distribution", font=dict(size=13, color="#94a3b8")),
        template="plotly_dark", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        height=300, margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(gridcolor=GRID_COLOR, ticksuffix="%"),
        yaxis=dict(gridcolor=GRID_COLOR), showlegend=False
    )
    st.plotly_chart(fig_ret, use_container_width=True)

with ra2:
    dd_series = (cum / cum.cummax() - 1) * 100
    fig_dd    = go.Figure()
    fig_dd.add_trace(go.Scatter(x=data['Close'].index[-len(dd_series):], y=dd_series,
                                name="Drawdown", line=dict(color=RED, width=1.5),
                                fill='tozeroy', fillcolor='rgba(252,129,129,0.1)'))
    fig_dd.update_layout(
        title=dict(text="Historical Drawdown", font=dict(size=13, color="#94a3b8")),
        template="plotly_dark", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        height=300, margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, ticksuffix="%"), showlegend=False
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# ── DATA TABLE ────────────────────────────────────────────────────────────────

st.markdown('<p class="section-header">Raw Market Data</p>', unsafe_allow_html=True)

display_cols = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in data.columns]
display_data = data[display_cols].tail(60).copy()
display_data['Date'] = display_data['Date'].dt.strftime('%Y-%m-%d')
for c in ['Open', 'High', 'Low', 'Close']:
    if c in display_data.columns:
        display_data[c] = display_data[c].map(lambda x: f"${x:,.2f}")
if 'Volume' in display_data.columns:
    display_data['Volume'] = display_data['Volume'].map(lambda x: f"{int(x):,}")

st.dataframe(display_data, use_container_width=True, hide_index=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1rem;color:#2d3748;font-size:0.75rem;letter-spacing:0.05em;">
    Market Lens &nbsp;|&nbsp; AI Stock Intelligence &nbsp;|&nbsp;
    LSTM + Transformer Ensemble &nbsp;|&nbsp;
    <em>Not financial advice.</em>
</div>
""", unsafe_allow_html=True)


