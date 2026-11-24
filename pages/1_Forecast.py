import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st

from fxviz import config, data, outputs

st.set_page_config(page_title="Forecast · Exchange Rate Visualizer", page_icon="🔮", layout="wide")


@st.cache_data(ttl=300)
def load_all():
    return (
        outputs.load_forecasts(),
        outputs.load_metrics(),
        outputs.load_run_info(),
        data.load_rates(),
    )


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


st.title("🔮 30-business-day forecast")

forecasts, metrics, info, rates = load_all()
if forecasts is None or rates.empty:
    st.info("No forecasts yet. Generate them with `python -m fxviz.pipeline` (see the README).")
    st.stop()

with st.sidebar:
    pair = st.selectbox("Currency", config.PAIRS)
    model_names = sorted(forecasts["model"].unique(), key=lambda m: (m != "naive", m))
    chosen = st.multiselect("Models", model_names, default=model_names)
    history_days = st.slider("History shown (business days)", 30, 500, 120, step=10)

history = rates[pair].dropna().iloc[-history_days:]
palette = pc.qualitative.Plotly

fig = go.Figure()
fig.add_trace(go.Scatter(x=history.index, y=history, name="Actual", line=dict(color="gray", width=2)))
for i, model in enumerate(chosen):
    g = forecasts[(forecasts["currency"] == pair) & (forecasts["model"] == model)]
    color = palette[(i + 1) % len(palette)]
    fig.add_trace(go.Scatter(x=g["date"], y=g["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=g["date"], y=g["lower"], line=dict(width=0), fill="tonexty",
            fillcolor=rgba(color, 0.18), name=f"{model} 80% interval", hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=g["date"], y=g["forecast"], name=model, line=dict(color=color, width=2, dash="dot")))

cutoff = history.index[-1].strftime("%Y-%m-%d")
fig.add_shape(type="line", x0=cutoff, x1=cutoff, y0=0, y1=1, yref="paper", line=dict(dash="dash", color="gray"))
fig.update_layout(
    hovermode="x unified",
    yaxis_title=f"{pair[:3]} per 1 EUR",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=30, b=10),
    height=520,
)
st.plotly_chart(fig)

# --- honest read-out ----------------------------------------------------------------------
st.subheader("How good is it?")
m = metrics[(metrics["currency"] == pair) & (metrics["model"].isin(chosen))] if metrics is not None else None
if m is not None and not m.empty:
    for _, row in m.iterrows():
        if row["model"] == "naive":
            continue
        verdict = "beats" if row["skill_vs_naive"] > 0 else "does **not** beat"
        st.markdown(
            f"- **{row['model']}** {verdict} the naive “no change” baseline on held-out data "
            f"(MAE skill {row['skill_vs_naive']:+.1%})."
        )
    st.caption("Exchange rates are very close to a random walk, so the naive baseline is the bar to clear.")

summary = (
    forecasts[(forecasts["currency"] == pair) & (forecasts["model"].isin(chosen))]
    .sort_values("date")
    .groupby("model")
    .tail(1)
    .rename(columns={"date": "last forecast day", "forecast": "forecast", "lower": "80% low", "upper": "80% high"})
    .drop(columns="currency")
)
st.dataframe(summary, hide_index=True)

with st.expander("Forecast data"):
    table = forecasts[(forecasts["currency"] == pair) & (forecasts["model"].isin(chosen))]
    st.dataframe(table, hide_index=True)
st.download_button(
    "📥 Download all forecasts (CSV)",
    forecasts.to_csv(index=False).encode(),
    file_name="forecasts.csv",
    mime="text/csv",
)

if info:
    st.caption(
        f"Data up to {info['data_end']} · generated {info['generated_at'][:10]} · "
        "Educational purposes only, not financial advice."
    )
