import streamlit as st
import pandas as pd

st.title("📊 Model Metrics Summary")

try:
    df_metrics = pd.read_csv("metricas_monedas.csv")
    st.dataframe(df_metrics.style.format({"MAE": "{:.6f}", "RMSE": "{:.6f}", "R2": "{:.4f}"}))
except FileNotFoundError:
    st.error("❌ metricas_monedas.csv not found. Run the training first.")
