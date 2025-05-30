# 💱 Exchange Rate Visualizer 📈

An interactive tool to explore and forecast currency exchange rates using historical data from the [Frankfurter API](https://www.frankfurter.app/). This app uses **Streamlit** for the web interface, **Plotly** for interactive charts, and **LSTM models (via TensorFlow/Keras)** for forecasting future exchange rates.

---

## ✨ Features

- 📅 Fetches historical exchange rates from 1999 to the present.
- 📊 Interactive Plotly charts to visualize exchange rate trends.
- 🤖 Time series forecasting with LSTM models (30-day prediction).
- 📥 Option to download CSV files of both historical data and forecasts.
- 📈 Displays model evaluation metrics: MAE, RMSE, and R².
- 🔁 Automatically updates models and overwrites old ones to ensure fresh predictions.

---

## 📦 Requirements

- Python 3.7 or higher
- `pip` (Python package installer)

### Recommended Environment Setup

You can use a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

bash
Copiar
Editar
pip install -r requirements.txt

```
🚀 How to Run Locally
Clone the repository:


git clone https://github.com/YOUR-USERNAME/exchange-rate-visualizer.git
cd exchange-rate-visualizer

Install dependencies:

pip install -r requirements.txt



Download and process historical data:

python train_model.py



Download and process historical data:

python fetch_data.py



Launch the Streamlit app:

streamlit run app.py


📂 Project Structure
bash
Copiar
Editar
.
├── app.py                        # Main Streamlit interface
├── train_model.py                # Script to download exchange rate data
├── train_lstm.py                # Trains LSTM models and saves forecasts
├── modelos_lstm/                # Directory where models are stored
│   ├── modelo_USD_EUR.keras
│   └── ...
├── data/
│   └── historical_rates.csv     # Data downloaded from Frankfurter API
├── prediccion_USD_EUR_30dias.csv  # Forecast output example
├── metricas_monedas.csv         # Evaluation metrics (MAE, RMSE, R2)
├── requirements.txt             # Python dependencies
└── README.md                    # This file



🛠️ Notes
Before retraining models with train_lstm.py, old models in modelos_lstm/ will be automatically overwritten.

Each model forecasts 30 days into the future based on the last 30-day sequence.

The app is intended for educational and experimental purposes; predictions are statistical estimates, not financial advice.

Compatible with Windows, macOS, and Linux environments where Python ≥ 3.7 is available.