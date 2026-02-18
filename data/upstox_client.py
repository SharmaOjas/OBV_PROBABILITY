import streamlit as st
import pandas as pd
import gzip
from upstox_client.rest import ApiException
import io

# -------------------------------------------------
# Upstox Instrument Mapping
# -------------------------------------------------
@st.cache_data(ttl=86400)
def get_upstox_instruments():
    url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with gzip.open(io.BytesIO(response.content), 'rt') as f:
                df = pd.read_csv(f)
            return df[df['exchange'].isin(['NSE_EQ', 'BSE_EQ', 'NSE_FO', 'NSE_INDEX'])]
    except Exception as e:
        st.error(f"Error fetching instrument list: {e}")
    return pd.DataFrame()


def get_instrument_key(instruments_df, ticker, is_index=False):
    if instruments_df.empty:
        return None

    if is_index:
        index_map = {
            "NIFTY 50": "NIFTY", "NIFTY BANK": "BANKNIFTY",
            "NIFTY FIN SERVICE": "FINNIFTY", "NIFTY MIDCAP SELECT": "MIDCPNIFTY",
        }
        search_symbol = index_map.get(ticker, ticker)
        futures = instruments_df[
            (instruments_df['exchange'] == 'NSE_FO') &
            (instruments_df['instrument_type'] == 'FUTIDX') &
            (instruments_df['tradingsymbol'].str.startswith(search_symbol))
            ].copy()
        if not futures.empty:
            futures['expiry'] = pd.to_datetime(futures['expiry'])
            return futures.sort_values('expiry').iloc[0]['instrument_key']

    clean_ticker = ticker.split('.')[0].upper()
    exchange = 'NSE_EQ' if ticker.endswith('.NS') else 'BSE_EQ' if ticker.endswith('.BO') else 'NSE_EQ'
    match = instruments_df[
        (instruments_df['tradingsymbol'].str.upper() == clean_ticker) &
        (instruments_df['exchange'] == exchange)
        ]
    return match.iloc[0]['instrument_key'] if not match.empty else None


def fetch_upstox_historical_data(instrument_key, interval, from_date, to_date, access_token):
    api_instance = upstox_client.HistoryApi()
    api_instance.api_client.configuration.access_token = access_token
    try:
        api_response = api_instance.get_historical_candle_data1(
            instrument_key, interval, to_date, from_date, "v2"
        )
        if api_response.status == "success" and api_response.data.candles:
            df = pd.DataFrame(
                api_response.data.candles,
                columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']
            )
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)
            df.sort_index(inplace=True)
            return df
    except ApiException as e:
        st.error(f"Upstox API Error for {instrument_key}: {e}")
    except Exception as e:
        st.error(f"General Error for {instrument_key}: {e}")
    return pd.DataFrame()

