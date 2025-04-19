import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import requests
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta  # สำหรับ indicators

# ----- CONFIG -----
TICKERS = ["NVDA", "AMZN", "RKLB", "TSM", "LLY", "AVGO", "HIMS", 
           "PLTR", "TMDX", "ASML", "ARQT", "V", "META", "ABBV"]
FORECAST_DAYS = 7
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
# ------------------

def download_data(ticker):
    end = dt.datetime.now()
    start = end - dt.timedelta(days=365*2)
    df = yf.download(ticker, start=start)
    df = df[['Close']].dropna()
    return df

def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df['macd_signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['ema50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    df['ema100'] = ta.trend.EMAIndicator(df['Close'], window=100).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()
    df = df.dropna()
    return df

def prepare_data(df):
    df = add_indicators(df)
    features = df[['Close', 'rsi', 'macd', 'macd_signal', 'ema50', 'ema100', 'ema200']]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    X, y = [], []
    for i in range(60, len(scaled) - FORECAST_DAYS):
        X.append(scaled[i-60:i])
        y.append(scaled[i:i+FORECAST_DAYS, 0])  # พยากรณ์ราคาปิด
    return np.array(X), np.array(y), scaler, df

def build_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(60, 7)))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(FORECAST_DAYS))
    model.compile(optimizer='adam', loss='mse')
    return model

def support_levels(series, days=[14, 30, 90]):
    levels = {}
    for d in days:
        levels[f"{d}d"] = series[-d:].min()
    return levels

def forecast_stock(ticker):
    df = download_data(ticker)
    X, y, scaler, df = prepare_data(df)
    model = build_model()
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    last_seq = df[-60:][['Close', 'rsi', 'macd', 'macd_signal', 'ema50', 'ema100', 'ema200']]
    last_scaled = scaler.transform(last_seq)
    forecast_scaled = model.predict(np.expand_dims(last_scaled, axis=0))[0]

    forecast = scaler.inverse_transform(
        np.concatenate([
            forecast_scaled.reshape(-1, 1),
            np.zeros((FORECAST_DAYS, 6))
        ], axis=1)
    )[:, 0]

    pred_test = model.predict(X)
    test_true = y[:, -1]
    test_pred = pred_test[:, -1]
    rmse = np.sqrt(mean_squared_error(test_true, test_pred))
    accuracy = 100 - (rmse / test_true.mean() * 100)

    return {
        "ticker": ticker,
        "forecast": forecast,
        "accuracy": accuracy,
        "last_price": df['Close'].iloc[-1],
        "support": support_levels(df['Close'])
    }

def send_to_discord(all_data):
    if not WEBHOOK_URL:
        print("No webhook set.")
        return

    msg = "**📊 Stock Forecast Report (Next 7 Days)**\n"
    for data in all_data:
        msg += f"\n📈 **{data['ticker']}**\n"
        msg += f"    ├ Accuracy: `{data['accuracy']:.2f}%`\n"
        msg += f"    ├ Forecast: {['$%.2f' % x for x in data['forecast']]}\n"
        msg += f"    ├ Last Close: `${data['last_price']:.2f}`\n"
        msg += f"    └ Support Levels: {', '.join([f'{k}: ${v:.2f}' for k, v in data['support'].items()])}\n"

    requests.post(WEBHOOK_URL, json={"content": msg})

def main():
    summary = []
    for ticker in TICKERS:
        try:
            summary.append(forecast_stock(ticker))
        except Exception as e:
            print(f"Error for {ticker}: {e}")
    send_to_discord(summary)

if __name__ == "__main__":
    main()
