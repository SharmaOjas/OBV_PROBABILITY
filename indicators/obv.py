import numpy as np

# -------------------------------------------------
# OBV + Divergence Logic
# -------------------------------------------------
def calculate_obv(df):
    df = df.copy()
    df['Change'] = df['Close'].diff()
    df['OBV_Vol'] = np.where(df['Change'] > 0, df['Volume'],
                             np.where(df['Change'] < 0, -df['Volume'], 0))
    df['OBV'] = df['OBV_Vol'].cumsum().fillna(0)
    return df

