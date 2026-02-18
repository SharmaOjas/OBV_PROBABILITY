from indicators.pivots import get_pivots

def detect_divergence(df, order=5):
    price_highs_idx, price_lows_idx = get_pivots(df['Close'], order)
    results = []

    if len(price_highs_idx) >= 2:
        for i in range(len(price_highs_idx[-5:]) - 1, 0, -1):
            peaks = price_highs_idx[-5:]
            ci, pi = peaks[i], peaks[i - 1]
            if df['Close'].iloc[ci] > df['Close'].iloc[pi] and df['OBV'].iloc[ci] < df['OBV'].iloc[pi]:
                results.append({
                    "Type": "Bearish",
                    "P1_Idx": pi, "P2_Idx": ci,
                    "P1_Date": df.index[pi], "P2_Date": df.index[ci],
                    "P1_Price": df['Close'].iloc[pi], "P2_Price": df['Close'].iloc[ci],
                    "P1_OBV": df['OBV'].iloc[pi], "P2_OBV": df['OBV'].iloc[ci],
                })
                break

    if len(price_lows_idx) >= 2:
        for i in range(len(price_lows_idx[-5:]) - 1, 0, -1):
            lows = price_lows_idx[-5:]
            ci, pi = lows[i], lows[i - 1]
            if df['Close'].iloc[ci] < df['Close'].iloc[pi] and df['OBV'].iloc[ci] > df['OBV'].iloc[pi]:
                results.append({
                    "Type": "Bullish",
                    "P1_Idx": pi, "P2_Idx": ci,
                    "P1_Date": df.index[pi], "P2_Date": df.index[ci],
                    "P1_Price": df['Close'].iloc[pi], "P2_Price": df['Close'].iloc[ci],
                    "P1_OBV": df['OBV'].iloc[pi], "P2_OBV": df['OBV'].iloc[ci],
                })
                break

    return results, price_highs_idx, price_lows_idx

