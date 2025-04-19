import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import os
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

# ----------------------- CONFIG -----------------------
TICKERS = ["NVDA", "AMZN", "RKLB", "TSM", "LLY", "AVGO", "HIMS", 
           "PLTR", "TMDX", "ASML", "ARQT", "V", "META", "ABBV"]
FORECAST_DAYS = 7
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
# ------------------------------------------------------

def download_data(ticker):
    end = dt.datetime.now()
    start = end - dt.timedelta(days=365*5)
    df = yf.download(ticker, start=start, end=end)
    df = df[["Close"]].dropna()
    return df

def prepare_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(60, len(scaled) - FORECAST_DAYS):
        X.append(scaled[i-60:i])
        y.append(scaled[i:i+FORECAST_DAYS])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(FORECAST_DAYS))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_stock(ticker):
    df = download_data(ticker)
    X, y, scaler = prepare_data(df)

    model = build_model()
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # พยากรณ์จากข้อมูลล่าสุด
    last_60 = df[-60:].values
    last_scaled = scaler.transform(last_60)
    last_scaled = np.reshape(last_scaled, (1, 60, 1))
    forecast_scaled = model.predict(last_scaled)[0]
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

    # ประเมินความแม่นยำจาก test set
    pred_test = model.predict(X)
    test_true = scaler.inverse_transform(y[:, -1].reshape(-1, 1))
    test_pred = scaler.inverse_transform(pred_test[:, -1].reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(test_true, test_pred))
    accuracy = 100 - (rmse / test_true.mean() * 100)

    return forecast, accuracy, df

def send_forecast_to_discord(summary):
    if not WEBHOOK_URL:
        print("No Discord Webhook URL.")
        return

    msg = "**📊 Stock Forecast Report (Next 7 Days)**\n"
    for item in summary:
        msg += f"\n📈 **{item['ticker']}**\n"
        msg += f"    ├ Accuracy: `{item['accuracy']:.2f}%`\n"
        msg += f"    ├ Forecast: {['$%.2f' % x for x in item['forecast']]}\n"
        msg += f"    └ Last Close: `${item['last_price']:.2f}`\n"

    requests.post(WEBHOOK_URL, json={"content": msg})

def main():
    summary = []
    for ticker in TICKERS:
        forecast, acc, df = forecast_stock(ticker)
        summary.append({
            "ticker": ticker,
            "forecast": forecast,
            "accuracy": acc,
            "last_price": df['Close'].iloc[-1]
        })
    send_forecast_to_discord(summary)

if __name__ == "__main__":
    main()
