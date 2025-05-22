import os
import time
import sys
import numpy as np
import pandas as pd
import logging
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, TimeDistributed, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ======= CONFIGURATION =======
csv_file = 'data/historical_rates.csv'
currencies = ['USD/EUR', 'AUD/EUR', 'GBP/EUR', 'CHF/EUR']
look_back = 1000
future_days = 30
save_models = True
models_folder = 'modelos_lstm'
metrics_file = 'metricas_monedas.csv'

# ======= LOGGING =======
logging.basicConfig(filename='training_lstm.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    filemode='w')

def log(msg):
    print(msg)
    logging.info(msg)

if save_models and not os.path.exists(models_folder):
    os.makedirs(models_folder)

# ======= FUNCTIONS =======
def create_sequences(data, look_back, future_steps):
    X, y = [], []
    for i in range(len(data) - look_back - future_steps):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back:i + look_back + future_steps])
    return np.array(X), np.array(y)

def build_model(input_shape, output_steps):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(128, return_sequences=False))
    model.add(RepeatVector(output_steps))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ======= LOAD DATA =======
df = pd.read_csv(csv_file, parse_dates=['Date'])
df.set_index('Date', inplace=True)

all_metrics = []

# ======= TRAINING LOOP =======
for currency in currencies:
    log(f"\n🔄 Training model for {currency}...")
    start = time.time()

    series = df[currency].dropna().values.reshape(-1, 1)
    if len(series) < look_back + future_days:
        log(f"❌ Not enough data for {currency}.")
        continue

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    X, y = create_sequences(scaled, look_back, future_days)
    y = y.reshape(y.shape[0], y.shape[1], 1)

    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    model = build_model((look_back, 1), future_days)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=300,
              batch_size=64,
              callbacks=callbacks,
              verbose=1)

    # ======= EVALUATION =======
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1, future_days)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, future_days)

    mae = np.mean([mean_absolute_error(t, p) for t, p in zip(y_true, y_pred)])
    rmse = np.mean([np.sqrt(mean_squared_error(t, p)) for t, p in zip(y_true, y_pred)])
    r2 = np.mean([r2_score(t, p) for t, p in zip(y_true, y_pred)])

    log(f"📊 MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.4f}")
    all_metrics.append({'Currency': currency, 'MAE': mae, 'RMSE': rmse, 'R2': r2})

    # ======= FINAL PREDICTION =======
    last_window = scaled[-look_back:].reshape(1, look_back, 1)
    prediction_scaled = model.predict(last_window, verbose=0).reshape(-1, 1)
    prediction = scaler.inverse_transform(prediction_scaled).flatten()

    start_date = df.index[-1] + timedelta(days=1)
    future_dates = pd.date_range(start_date, periods=future_days, freq='D')
    df_pred = pd.DataFrame({f'Prediction {currency}': prediction}, index=future_dates)
    df_pred.to_csv(f'prediccion_{currency.replace("/", "_")}_30dias.csv')
    log(f"💾 Saved prediction: prediccion_{currency.replace('/', '_')}_30dias.csv")

    # ======= SAVE MODEL =======
    if save_models:
        model_path = os.path.join(models_folder, f'modelo_{currency.replace("/", "_")}.keras')
        model.save(model_path)
        log(f"💾 Model saved: {model_path}")

    log(f"✅ Completed {currency} in {(time.time() - start)/60:.2f} minutes")
    sys.stdout.flush()

# ======= SAVE METRICS =======
pd.DataFrame(all_metrics).to_csv(metrics_file, index=False)
log(f"📁 Metrics saved to {metrics_file}")
