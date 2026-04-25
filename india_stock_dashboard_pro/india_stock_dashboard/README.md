# 🇮🇳 India Stock Market Dashboard
### AI/ML Project — Python · Streamlit · scikit-learn · yfinance

---

## 📋 Features

| Section | What it does |
|---|---|
| 🏠 Market Overview | Live Nifty 50 & Sensex index cards, price chart, stock snapshot table |
| 📊 Stock Analysis | OHLC chart + SMA/EMA/Bollinger Bands, MACD, RSI, Volume — all in one view |
| 🤖 ML Prediction | Trains 3 ML models (Linear Regression, Random Forest, Gradient Boosting), shows best model, 30-day forecast + feature importance |
| 🔥 Heatmap & Sectors | Color-coded heatmap of Nifty 50 stocks, sector performance bar chart + pie chart |
| 💡 Insights & Signals | Buy/Sell/Hold signal engine using RSI + MACD + SMA + Bollinger, bulk signal scan |
| 💰 Portfolio Simulator | SIP calculator, Lumpsum calculator, Portfolio builder with Sharpe ratio & Max Drawdown |

---

## 🚀 How to Execute (Step-by-Step)

### Step 1 — Install Python
Download Python 3.10 or 3.11 from https://www.python.org/downloads/
✅ Check "Add Python to PATH" during installation.

### Step 2 — Open Terminal / Command Prompt
- **Windows**: Press `Win + R`, type `cmd`, press Enter
- **Mac/Linux**: Open Terminal

### Step 3 — Navigate to project folder
```bash
cd path/to/india_stock_dashboard
# Example on Windows: cd C:\Users\YourName\Downloads\india_stock_dashboard
# Example on Mac:     cd ~/Downloads/india_stock_dashboard
```

### Step 4 — (Recommended) Create a virtual environment
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 5 — Install dependencies
```bash
pip install -r requirements.txt
```
This installs: streamlit, yfinance, pandas, numpy, matplotlib, scikit-learn, plotly

### Step 6 — Run the app
```bash
streamlit run app.py
```

### Step 7 — Open in browser
The terminal will show a URL like:
```
Local URL: http://localhost:8501
```
Open that URL in your browser. The dashboard loads automatically!

---

## 📁 File Structure
```
india_stock_dashboard/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🤖 ML Models Used

| Model | Library | Use |
|---|---|---|
| Linear Regression | scikit-learn | Baseline price prediction |
| Random Forest | scikit-learn | Non-linear pattern learning |
| Gradient Boosting | scikit-learn | Best accuracy, feature importance |
| MinMaxScaler | scikit-learn | Feature normalization |

### Features fed to ML models:
- SMA 20, SMA 50, EMA 12, EMA 26
- RSI (14), MACD, ATR
- Bollinger Bands (Upper/Lower)
- Volume, Lag prices (1,2,3,5 days)
- Price Range, Price Change, Daily Return, Volatility

---

## 📊 Technical Indicators Implemented
- **SMA** (20, 50, 200-day Simple Moving Average)
- **EMA** (12, 26-day Exponential Moving Average)
- **MACD** + Signal Line + Histogram
- **RSI** (14-day Relative Strength Index)
- **Bollinger Bands** (20-day, ±2 SD)
- **ATR** (14-day Average True Range)
- **OBV** (On-Balance Volume)
- **52-Week High/Low**
- **Annualised Volatility**

---

## ⚠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| No live data / empty charts | yfinance may be rate-limited; synthetic data is used as fallback |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |
| Slow loading | First load trains ML models; subsequent loads use cache (5 min TTL) |

---

## 📌 Disclaimer
This dashboard is built for **educational purposes** as part of an AI/ML course project.
Data is sourced from Yahoo Finance (NSE). This is **not financial advice**.
