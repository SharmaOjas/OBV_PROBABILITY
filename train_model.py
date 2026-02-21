# train_model.py
import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
from indicators import calculate_obv, compute_rsi, detect_divergence_all
import requests
from io import StringIO


# NSE stock list (start small — Not using 500 yet).
def fetch_nse500():
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    df = pd.read_csv(StringIO(r.text))
    return df["Symbol"].tolist()

symbols = fetch_nse500()
stocks = [s + ".NS" for s in symbols[:200]]

def download_data(stocks, period="2y"):
    all_data = {}

    for symbol in tqdm(stocks):
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
        # Flatten columns if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)        
        # Basic sanity filter
        if df is not None and len(df) > 200:
            df.dropna(inplace=True)
            all_data[symbol] = df

    return all_data

if __name__ == "__main__":
    print("Downloading data...")
    data = download_data(stocks)

    print(f"Downloaded data for {len(data)} stocks.")

    # Print one sample to verify
    for symbol, df in data.items():
        print(f"\nSample for {symbol}:")
        print(df.tail())
        break
all_labels = []



def create_label(df, idx, div_type, horizon=20, threshold=0.07):
    if idx + horizon >= len(df):
        return None  # not enough future data

    entry_price = df["Close"].iloc[idx]
    future_prices = df["Close"].iloc[idx + 1: idx + horizon + 1]

    if div_type == "Bullish":
        max_future = future_prices.max()
        return 1 if (max_future - entry_price) / entry_price >= threshold else 0

    else:  # Bearish
        min_future = future_prices.min()
        return 1 if (entry_price - min_future) / entry_price >= threshold else 0






for symbol, df in data.items():
    df = calculate_obv(df)
    divergences = detect_divergence_all(df)

    for div_type, idx in divergences:
        label = create_label(df, idx, div_type)
        if label is not None:
            all_labels.append(label)

print("Total samples:", len(all_labels))
print("Success rate:", sum(all_labels) / len(all_labels))


#just to check Class balance im adding this:
print("Wins:", sum(all_labels))
print("Losses:", len(all_labels) - sum(all_labels))






#the issue was that the success rate was 91% which is too high maybe overfitting
bull_wins = 0
bull_total = 0
bear_wins = 0
bear_total = 0

for symbol, df in data.items():
    df = calculate_obv(df)
    divergences = detect_divergence_all(df)

    for div_type, idx in divergences:
        label = create_label(df, idx, div_type)
        if label is None:
            continue

        if div_type == "Bullish":
            bull_total += 1
            bull_wins += label
        else:
            bear_total += 1
            bear_wins += label

print("Bullish success:", bull_wins / bull_total)
print("Bearish success:", bear_wins / bear_total)