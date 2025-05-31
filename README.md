# 💱 Exchange Rate Visualizer 📈

An interactive tool to explore and forecast currency exchange rates using historical data from the [Frankfurter API](https://www.frankfurter.app/). This app uses **Streamlit** for the web interface, **Plotly** for interactive charts, and **LSTM models (via TensorFlow/Keras)** for forecasting future exchange rates.

---

## 🌐 Objective

To provide users with an interactive, data-driven tool for analyzing historical exchange rate trends and producing short-term forecasts using deep learning. The system is designed to be educational, reproducible, and easy to deploy in local or production environments.

---

## ✨ Features

- 📅 Fetches historical exchange rates from 1999 to the present.
- 📊 Interactive Plotly charts to visualize exchange rate trends.
- 🤖 Time series forecasting with LSTM models (30-day prediction).
- 📥 Option to download CSV files of both historical data and forecasts.
- 📈 Displays model evaluation metrics: MAE, RMSE, and R².
- 🔁 Automatically updates models and overwrites old ones to ensure fresh predictions.

---

## 🧠 Technologies Used

- 🐍 Python 3.10.11
- 🧠 TensorFlow / Keras for LSTM modeling
- 📊 Plotly for interactive data visualization
- 🖼️ Streamlit for the web interface
- 📈 Pandas, NumPy for data processing
- 🔄 Python scripting for ETL and model automation
- 🗃️ Modular project structure ready for containerization and future automation

---

## 📦 Requirements

- Python 3.7 or higher
- `pip` (Python package installer)

### Recommended Environment Setup

You can use a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run Locally

### 1. Clone the repository:
```bash
git clone https://github.com/JM181998/exchange-rate-visualizer.git
cd exchange-rate-visualizer
```

### 2. Download and process historical data:
```bash
python train_model.py
```

### 3. Launch the Streamlit app:
```bash
streamlit run app.py
```

---

## 📂 Project Structure


| Folder / File                    | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `app.py`                         | Streamlit app interface to visualize historical data and forecasts.        |
| `fetch_data.py`                  | Script to download historical exchange rate data from Frankfurter API.     |
| `train_lstm.py`                  | Trains LSTM models for selected currency pairs and saves predictions.      |
| `modelos_lstm/`                  | Folder where trained LSTM models are stored (`.keras` format).             |
| `data/`                          | Contains processed historical data used for training and visualization.    |
| `prediccion_USD_EUR_30dias.csv` | Example CSV file with 30-day forecast for USD to EUR.                      |
| `metricas_monedas.csv`          | CSV file with model evaluation metrics: MAE, RMSE, R².                      |
| `requirements.txt`              | List of required Python libraries.                                         |
| `README.md`                     | Project documentation (this file).       

---

## 🛠️ Notes

- ✅ Before retraining models with train_lstm.py, old models in modelos_lstm/ will be automatically overwritten.
- 🔮 Each model forecasts 30 days into the future based on the last 30-day sequence.
- ⚠️ This app is intended for educational and experimental purposes; predictions are statistical estimates, not financial advice.
- 💻 Compatible with Windows, macOS, and Linux environments where Python ≥ 3.7 is available.

---

## 📬 Contact

Feel free to open an issue or fork the repository if you'd like to contribute or provide feedback!

📌 Follow me in [LinkedIn](https://www.linkedin.com/in/juanma-fuentes/)