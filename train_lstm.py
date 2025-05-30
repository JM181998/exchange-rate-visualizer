import os
import time
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_dataset(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

currencies = ['USD/EUR', 'AUD/EUR', 'GBP/EUR', 'CHF/EUR']
n_steps = 30
epochs = 80
batch_size = 32
future_days = 30
model_dir = "modelos_lstm"
os.makedirs(model_dir, exist_ok=True)

for f in os.listdir(model_dir):
    if f.endswith(".keras"):
        os.remove(os.path.join(model_dir, f))

metrics = []

df = pd.read_csv("data/historical_rates.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

for currency in currencies:
    print(f"\n▶ Training model for {currency}")
    start_time = time.time()

    series = df[currency].dropna().values.reshape(-1, 1)

    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)

    X, y = create_dataset(series_scaled, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=False, input_shape=(n_steps, 1)),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5)

    model.fit(X, y, epochs=epochs, batch_size=batch_size,
              validation_split=0.2, callbacks=[early_stop, reduce_lr], verbose=0)

    last_seq = series_scaled[-n_steps:]
    forecast_scaled = []
    current_seq = last_seq.reshape(1, n_steps, 1)

    for _ in range(future_days):
        next_val = model.predict(current_seq, verbose=0)[0, 0]
        forecast_scaled.append(next_val)
        current_seq = np.append(current_seq[:, 1:, :], [[[next_val]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))

    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days)
    df_pred = pd.DataFrame(forecast, index=forecast_index, columns=[currency])
    df_pred.to_csv(f"prediccion_{currency.replace('/', '_')}_30dias.csv")
    print(f"💾 Saved prediction: prediccion_{currency.replace('/', '_')}_30dias.csv")

    model.save(os.path.join(model_dir, f"modelo_{currency.replace('/', '_')}.keras"))
    print(f"💾 Model saved: {model_dir}/modelo_{currency.replace('/', '_')}.keras")

    real_eval = series[-future_days:].flatten()
    pred_eval = forecast[:len(real_eval)].flatten()

    mae = mean_absolute_error(real_eval, pred_eval)
    rmse = mean_squared_error(real_eval, pred_eval) ** 0.5
    r2 = r2_score(real_eval, pred_eval)
    metrics.append([currency, mae, rmse, r2])

    print(f"📊 MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.4f}")
    print(f"✅ Completed {currency} in {time.time() - start_time:.2f} seconds")

metrics_df = pd.DataFrame(metrics, columns=["Currency", "MAE", "RMSE", "R2"])
metrics_df.to_csv("metricas_monedas.csv", index=False)
print("📁 Metrics saved to metricas_monedas.csv")
