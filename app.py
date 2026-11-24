import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fxviz import config, data

st.set_page_config(page_title="Exchange Rate Visualizer", page_icon="💱", layout="wide")


@st.cache_data(ttl=6 * 3600, show_spinner="Updating exchange rates (ECB via Frankfurter)…")
def get_rates():
    """Incrementally refresh the cached dataset; fall back to whatever is on disk."""
    try:
        return data.update_rates(), None
    except data.DataUnavailable as exc:
        return data.load_rates(), str(exc)


PRESETS = {"1M": 30, "6M": 182, "1Y": 365, "5Y": 1826, "10Y": 3652, "Max": None}

st.title("💱 Exchange Rate Visualizer")
st.caption("Euro reference rates published by the European Central Bank, served by the Frankfurter API.")

rates, error = get_rates()
if rates.empty:
    st.error(f"No exchange-rate data available. {error or ''}")
    st.stop()
if error:
    st.warning(f"Could not refresh the data ({error}). Showing the last saved dataset.")
if not data.is_daily(rates):
    st.info("The saved dataset is weekly (legacy). It will be replaced by daily data on the next successful update.")

with st.sidebar:
    st.header("Options")
    selected = st.multiselect("Currencies", config.PAIRS, default=list(config.PAIRS))
    period = st.radio("Period", list(PRESETS), index=3, horizontal=True)
    mode = st.radio("Scale", ["Rate", "Rebased to 100"], help="Rebasing makes currencies with different levels comparable.")

if not selected:
    st.info("Select at least one currency in the sidebar.")
    st.stop()

last_date = rates.index.max()
days = PRESETS[period]
view = rates[selected] if days is None else rates.loc[rates.index >= last_date - pd.Timedelta(days=days), selected]

# --- headline numbers -------------------------------------------------------------------
columns = st.columns(len(selected))
for col, pair in zip(columns, selected):
    series = rates[pair].dropna()
    latest = series.iloc[-1]
    year_ago = series.asof(series.index[-1] - pd.Timedelta(days=365))
    delta = f"{(latest / year_ago - 1) * 100:+.1f}% vs 1 year ago" if pd.notna(year_ago) else None
    col.metric(f"1 EUR in {pair[:3]}", f"{latest:.4f}", delta, delta_color="off")
st.caption(f"Latest observation: {last_date:%d %b %Y}")

# --- chart ------------------------------------------------------------------------------
plot_data = view.copy()
if mode == "Rebased to 100":
    plot_data = plot_data / plot_data.apply(lambda s: s.dropna().iloc[0]) * 100

fig = go.Figure()
for pair in selected:
    fig.add_trace(
        go.Scatter(
            x=plot_data.index,
            y=plot_data[pair],
            mode="lines",
            name=pair,
            hovertemplate="%{y:.4f}",
        )
    )
fig.update_layout(
    hovermode="x unified",
    yaxis_title="Units per 1 EUR" if mode == "Rate" else "Index (start = 100)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=30, b=10),
    height=520,
)
st.plotly_chart(fig)

with st.expander("Data table"):
    st.dataframe(view.sort_index(ascending=False))
st.download_button(
    "📥 Download selection as CSV",
    view.to_csv(index_label="Date").encode(),
    file_name="exchange_rates.csv",
    mime="text/csv",
)
st.caption("`USD/EUR` means: how many USD you get for 1 EUR. For educational purposes only, not financial advice.")
