import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.title("🔮 Forecast Viewer (30-Day LSTM Predictions)")

monedas = ['USD/EUR', 'AUD/EUR', 'GBP/EUR', 'CHF/EUR']
dias_futuro = 30
dias_historico = 365

try:
    df_hist = pd.read_csv("data/historical_rates.csv", parse_dates=["Date"])
    df_hist.set_index("Date", inplace=True)
except:
    st.error("❌ Could not load historical_rates.csv. Run the main page first.")
    st.stop()

for moneda in monedas:
    st.subheader(f"{moneda}")

    if moneda not in df_hist.columns:
        st.warning(f"No data for {moneda}")
        continue

    serie = df_hist[moneda].dropna().values[-dias_historico:]

    pred_file = f"prediccion_{moneda.replace('/', '_')}_30dias.csv"
    if not os.path.exists(pred_file):
        st.warning(f"Prediction file not found: {pred_file}")
        continue

    df_pred = pd.read_csv(pred_file, index_col=0, parse_dates=True)
    pred = df_pred.iloc[:, 0].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(-dias_historico, 0),
        y=serie,
        name="Historical",
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(0, dias_futuro),
        y=pred,
        name="Forecast",
        line=dict(color='orange')
    ))
    fig.add_vline(x=0, line=dict(dash='dash', color='gray'))
    fig.update_layout(
        title=f"{moneda} Forecast",
        xaxis_title="Days from Today",
        yaxis_title="Rate",
        legend_title="",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
