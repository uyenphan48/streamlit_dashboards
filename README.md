# Python Stock Analysis
This web app includes 4 dashboards: [➡️ Go to web app](https://share.streamlit.io/uyenphan48/streamlit_dashboards/main/stock_dashboards.py)

### 1. Overview Charts
Imports charts from finviz.com with closing price in the last 6 months for the interested stocks. It gives an overview of the stock prices and trends.

### 2. Stock Analysis
- Analysis on daily returns, correlations, clustering of the selected stocks. 
- Plots of stock patterns (SMA - Simple Moving Average, and Bollinger Bands).
- Risk Analysis using Monte Carlo simulations and Value at Risk.
Raw data is imported from Yahoo Finance and then used for the analysis.

### 4. Stock Prediction
Imports raw data from Yahoo Finance and forecasts price for the selected stocks and time.

### 5. Porfolio Tracker
- Imports transaction history (using Paper Trading) from web broker ([Alpaca](https://alpaca.markets/)) using API .
- Shows an overview of porforlio values, profit and lost.
- Portforlio breakdown by stock quantity and cost.
- This was developed for my personal use (using fixed API keys). For future, will improve it for other people use.

### An overview of the app:
https://user-images.githubusercontent.com/95778324/151162960-e96e9b32-3765-42ad-9905-a42524a81a2c.mp4

