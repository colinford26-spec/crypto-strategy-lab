# Crypto Strategy Lab

A simple local Streamlit application for downloading crypto OHLCV data and testing trading strategies. This is a backtesting/research tool, not financial advice or a live trading system.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Data is saved in `data/` and results can be downloaded as CSV files.

## Features

- Download OHLCV candles from a CCXT-supported exchange.
- Built-in SMA crossover and RSI strategies.
- Fees, slippage, position sizing, stop-loss and take-profit.
- Equity curve, drawdown, trade log and CSV export.
- No API keys are needed for public market-data endpoints.

## Notes

- Exchange history and available timeframes vary. One-minute, two-year history may not be available from every exchange.
- Test each strategy out-of-sample and compare it with buy-and-hold.
- This app does not place live orders.
