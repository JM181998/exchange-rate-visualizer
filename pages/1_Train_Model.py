import streamlit as st
import pandas as pd

st.title("🧠 Model Training Summary")
st.markdown("""
This page explains how the LSTM models were trained using historical exchange rate data.

- Input: Last **1000** days of data
- Forecast: **30** days into the future
- Models: One per currency (USD, GBP, AUD, CHF)
- Architecture: LSTM with 2 layers, dropout, TimeDistributed
- Optimizer: Adam with learning rate reduction & early stopping

📁 Trained models are saved in the `modelos_lstm/` folder.
📄 Prediction CSVs in root directory as: `prediccion_USD_EUR_30dias.csv`, etc.
""")

if st.button("⚠️ Train Model (Advanced)"):
    st.warning("⚠️ Training may take a long time (minutes or more). Not recommended unless necessary.")
    st.info("You can also train offline by running the training script locally.")

    # O puedes usar subprocess si lo integras como script:
    # import subprocess
    # subprocess.run(["python", "src/train_lstm.py"])
