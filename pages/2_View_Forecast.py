import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

st.title("🔮 Forecast Viewer (30-Day LSTM Predictions)")

# Definimos las monedas y los días que usaremos
currencies = ['USD/EUR', 'AUD/EUR', 'GBP/EUR', 'CHF/EUR']
dias_futuro = 30
dias_historico = 365

# Cargar histórico
try:
    df_hist = pd.read_csv("data/historical_rates.csv", parse_dates=["Date"])
    df_hist.set_index("Date", inplace=True)
except Exception as e:
    st.error(f"❌ Could not load historical_rates.csv: {e}")
    st.stop()

for currency in currencies:
    st.subheader(f"{currency}")

    if currency not in df_hist.columns:
        st.warning(f"No data found for {currency}")
        continue

    serie = df_hist[currency].dropna().values[-dias_historico:]

    pred_file = f"prediccion_{currency.replace('/', '_')}_30dias.csv"
    if not os.path.exists(pred_file):
        st.warning(f"⚠️ Prediction file not found: {pred_file}")
        continue

    # Cargar predicción
    df_pred = pd.read_csv(pred_file, index_col=0, parse_dates=True)
    pred = df_pred.iloc[:, 0].values

    # Crear figura
    fig = go.Figure()

    # Histórico
    fig.add_trace(go.Scatter(
        x=np.arange(-dias_historico, 0),
        y=serie,
        name="Historical",
        line=dict(color='blue')
    ))

    # Predicción
    fig.add_trace(go.Scatter(
        x=np.arange(0, dias_futuro),
        y=pred,
        name="Forecast",
        line=dict(color='orange')
    ))

    # Línea vertical divisoria
    fig.add_vline(x=0, line=dict(dash='dash', color='gray'))

    fig.update_layout(
        title=f"{currency} Forecast (Next 30 Days)",
        xaxis_title="Days from Today",
        yaxis_title="Exchange Rate",
        legend_title="",
        template="plotly_dark",  # Puedes cambiar a "plotly_white" si usas tema claro
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 View prediction data"):
        st.dataframe(df_pred)

    st.download_button(
        label=f"📥 Download {currency} Forecast CSV",
        data=df_pred.to_csv().encode(),
        file_name=pred_file,
        mime="text/csv"
    )
