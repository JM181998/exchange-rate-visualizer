import requests 
import plotly.graph_objects as go
import datetime
import pandas as pd

def fetch_exchange_rates():
    end_date = datetime.date.today().isoformat()
    url = f'https://api.frankfurter.app/1999-01-01..{end_date}?base=EUR&symbols=USD,AUD,GBP,CHF'
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code: {response.status_code}")
    
    data = response.json()
    dates, usd, aud, gbp, chf = [], [], [], [], []

    for date_str, values in data['rates'].items():
        dates.append(datetime.datetime.strptime(date_str, "%Y-%m-%d"))
        usd.append(values['USD'])
        aud.append(values['AUD'])
        gbp.append(values['GBP'])
        chf.append(values['CHF'])

    df = pd.DataFrame({
        'Date': dates,
        'USD/EUR': usd,
        'AUD/EUR': aud,
        'GBP/EUR': gbp,
        'CHF/EUR': chf
    })

    df.to_csv('data/historical_rates.csv', index=False)
    return df

def plot_exchange_rates(df):
    fig = go.Figure()
    currencies = ['USD/EUR', 'AUD/EUR', 'GBP/EUR', 'CHF/EUR']
    colors = ['blue', 'green', 'red', 'purple']

    for currency, color in zip(currencies, colors):
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df[currency], mode='lines', name=currency,
            hovertemplate=f"1 EUR = %{{y:.4f}} {currency[:3]}<br>Date: %{{x|%Y-%m-%d}}",
            line=dict(color=color)
        ))

    fig.update_layout(
        title="Exchange Rate Trends (EUR to Others)",
        xaxis_title="Date",
        yaxis_title="Exchange Rate",
        template="plotly_dark",
        hovermode="x unified"
    )

    fig.show()

if __name__ == "__main__":
    df = fetch_exchange_rates()
    plot_exchange_rates(df)

