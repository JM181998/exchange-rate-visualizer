import streamlit as st
import pandas as pd
import os
import subprocess

# Definir lista de monedas y rutas
currencies = ['USD/EUR', 'AUD/EUR', 'GBP/EUR', 'CHF/EUR']
models_folder = "modelos_lstm"

st.title("🧠 Model Training Summary")
st.markdown("""
This page explains how the LSTM models were trained using historical exchange rate data.

- Input: Last **1000** days of data  
- Forecast: **30** days into the future  
- Models: One per currency (USD, GBP, AUD, CHF)  
- Architecture: LSTM with 2 layers, dropout, TimeDistributed  
- Optimizer: Adam with learning rate reduction & early stopping  

📁 Trained models are saved in the `modelos_lstm/` folder.  
📄 Forecast CSVs are saved in the root directory as: `prediccion_USD_EUR_30dias.csv`, etc.
""")

if st.button("⚠️ Train Model (Advanced)"):
    st.warning("⚠️ Training may take a long time (minutes or more). Not recommended unless necessary.")
    st.info("You can also train offline by running the training script locally (`train_lstm.py`).")
    with st.spinner("Training models..."):
        result = subprocess.run(["python", "train_lstm.py"], capture_output=True, text=True)
        if result.returncode != 0:
            st.error("❌ Training script failed:")
            st.code(result.stderr)
        else:
            st.success("✅ Model training completed.")
            st.code(result.stdout)

st.subheader("📁 Download Results")

for currency in currencies:
    pred_file = f"prediccion_{currency.replace('/', '_')}_30dias.csv"
    model_file = os.path.join(models_folder, f"modelo_{currency.replace('/', '_')}.keras")

    if os.path.exists(pred_file):
        with open(pred_file, "rb") as f:
            st.download_button(
                label=f"📄 Download forecast CSV: {currency}",
                data=f,
                file_name=os.path.basename(pred_file),
                mime="text/csv"
            )
    else:
        st.warning(f"❌ Forecast file not found: {pred_file}")

    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            st.download_button(
                label=f"💾 Download model file: {currency}",
                data=f,
                file_name=os.path.basename(model_file),
                mime="application/octet-stream"
            )
    else:
        st.warning(f"❌ Model file not found: {model_file}")
