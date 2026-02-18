import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def calculate_obv(df):
    df = df.copy()
    df['Change'] = df['Close'].diff()
    df['OBV_Vol'] = np.where(df['Change'] > 0, df['Volume'],
                             np.where(df['Change'] < 0, -df['Volume'], 0))
    df['OBV'] = df['OBV_Vol'].cumsum().fillna(0)
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=series.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=series.index)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def get_pivots(data, order=5):
    return (
        argrelextrema(data.values, np.greater, order=order)[0],
        argrelextrema(data.values, np.less, order=order)[0],
    )


def detect_divergence(df, order=5):
    """
    Detects divergence by syncing Price Pivots with OBV timestamps.
    Returns a list of divergence dictionaries.
    """
    price_highs_idx, price_lows_idx = get_pivots(df['Close'], order)
    results = []

    # --- Bearish Divergence (Price Higher High, OBV Lower High) ---
    if len(price_highs_idx) >= 2:
        recent_peaks = price_highs_idx[-5:]  # Check recent 5 peaks
        for i in range(len(recent_peaks) - 1, 0, -1):
            curr_idx = recent_peaks[i]
            prev_idx = recent_peaks[i - 1]

            if df['Close'].iloc[curr_idx] > df['Close'].iloc[prev_idx]:  # Price HH
                if df['OBV'].iloc[curr_idx] < df['OBV'].iloc[prev_idx]:  # OBV LH
                    results.append({
                        "Type": "Bearish",
                        "P1_Idx": prev_idx, "P2_Idx": curr_idx,
                        "P1_Date": df.index[prev_idx], "P2_Date": df.index[curr_idx],
                        "P1_Price": df['Close'].iloc[prev_idx], "P2_Price": df['Close'].iloc[curr_idx],
                        "P1_OBV": df['OBV'].iloc[prev_idx], "P2_OBV": df['OBV'].iloc[curr_idx]
                    })
                    break  # Stop after finding the most recent one

    # --- Bullish Divergence (Price Lower Low, OBV Higher Low) ---
    if len(price_lows_idx) >= 2:
        recent_lows = price_lows_idx[-5:]
        for i in range(len(recent_lows) - 1, 0, -1):
            curr_idx = recent_lows[i]
            prev_idx = recent_lows[i - 1]

            if df['Close'].iloc[curr_idx] < df['Close'].iloc[prev_idx]:  # Price LL
                if df['OBV'].iloc[curr_idx] > df['OBV'].iloc[prev_idx]:  # OBV HL
                    results.append({
                        "Type": "Bullish",
                        "P1_Idx": prev_idx, "P2_Idx": curr_idx,
                        "P1_Date": df.index[prev_idx], "P2_Date": df.index[curr_idx],
                        "P1_Price": df['Close'].iloc[prev_idx], "P2_Price": df['Close'].iloc[curr_idx],
                        "P1_OBV": df['OBV'].iloc[prev_idx], "P2_OBV": df['OBV'].iloc[curr_idx]
                    })
                    break

    return results, price_highs_idx, price_lows_idx

