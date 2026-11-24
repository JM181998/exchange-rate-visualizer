import plotly.graph_objects as go
import streamlit as st

from fxviz import config, outputs

st.set_page_config(page_title="Metrics · Exchange Rate Visualizer", page_icon="📊", layout="wide")


@st.cache_data(ttl=300)
def load_all():
    return outputs.load_metrics(), outputs.load_horizon_errors(), outputs.load_run_info()


st.title("📊 Model metrics")

metrics, by_horizon, info = load_all()
if metrics is None:
    st.info("No metrics yet. Generate them with `python -m fxviz.pipeline` (see the README).")
    st.stop()

if info:
    st.markdown(
        f"Backtest on the last **{info['test_size_business_days']} business days** of data "
        f"(about {info['test_size_business_days'] // 250} years), forecasting "
        f"**{info['horizon_business_days']} business days** ahead from a new origin every "
        f"{info['origin_step_business_days']} days. Models never see data after their forecast origin."
    )

table = metrics.rename(
    columns={
        "currency": "Currency", "model": "Model", "mae": "MAE", "rmse": "RMSE",
        "mape": "MAPE %", "skill_vs_naive": "Skill vs naive", "n_origins": "Origins",
    }
)
st.dataframe(
    table.style.format(
        {"MAE": "{:.5f}", "RMSE": "{:.5f}", "MAPE %": "{:.2f}", "Skill vs naive": "{:+.1%}"}
    ),
    hide_index=True,
)
st.caption(
    "**Skill vs naive** = 1 − MAE / MAE of the “no change” forecast. Positive means the model "
    "beats the baseline, zero means it ties, negative means a simple flat line is better."
)

st.subheader("Error by forecast horizon")
pair = st.selectbox("Currency", config.PAIRS)
fig = go.Figure()
for model, g in by_horizon[by_horizon["currency"] == pair].groupby("model"):
    fig.add_trace(go.Scatter(x=g["horizon"], y=g["mae"], mode="lines", name=model))
fig.update_layout(
    xaxis_title="Business days ahead",
    yaxis_title="Mean absolute error",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig)
