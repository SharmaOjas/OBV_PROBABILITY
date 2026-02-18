import numpy as np
import pandas as pd


def calc_log_returns(close: np.ndarray) -> np.ndarray:
    c = np.array(close, dtype=float)
    lr = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        if c[i - 1] > 0 and c[i] > 0:
            lr[i] = np.log(c[i] / c[i - 1])
    return lr


def calc_pct_change(values: np.ndarray) -> np.ndarray:
    v = np.array(values, dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(1, len(v)):
        if not np.isnan(v[i - 1]) and v[i - 1] != 0:
            out[i] = (v[i] - v[i - 1]) * 100.0 / v[i - 1]
    return out


def calc_sma(arr: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        window_vals = arr[i - window + 1:i + 1]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) > 0:
            result[i] = np.mean(valid)
    return result


def calc_volume_weighted_garch(
        log_returns, volumes, window=20,
        omega=0.0001, alpha=0.15, beta=0.80, gamma=0.4,
):
    n = len(log_returns)
    if n == 0 or len(volumes) != n:
        return np.array([])

    volatilities = np.full(n, np.nan)
    avg_volume = np.nanmean(np.maximum(volumes, 0))
    if avg_volume <= 0:
        avg_volume = 1.0
    norm_vols = np.maximum(volumes, 0) / avg_volume

    init_slice = log_returns[:min(window, n)]
    valid_init = init_slice[np.isfinite(init_slice)]
    if len(valid_init) == 0:
        return volatilities
    uncond_var = np.clip(np.mean(valid_init ** 2), 0.0001, 1.0)
    prev_variance = uncond_var
    prev_sq_return = uncond_var

    for i in range(n):
        cur_ret = log_returns[i] if np.isfinite(log_returns[i]) else 0.0
        sq_ret = cur_ret ** 2
        vol_ratio = np.clip(norm_vols[i] if np.isfinite(norm_vols[i]) else 1.0, 0.1, 10.0)
        core_var = omega + alpha * prev_sq_return + beta * prev_variance
        vol_term = np.tanh(gamma * np.log(vol_ratio))
        new_var = core_var * (1.0 + vol_term)
        if not np.isfinite(new_var) or new_var < 0:
            new_var = prev_variance
        new_var = np.clip(new_var, 0.0001, 1.0)
        volatilities[i] = np.sqrt(new_var) * np.sqrt(252) * 100.0
        prev_variance = new_var
        prev_sq_return = sq_ret

    return volatilities


def calc_yang_zhang_vol(open_, high, low, close, volumes=None, window=10):
    n = len(close)
    if n < window:
        return np.full(n, np.nan)

    open_ = np.array(open_, dtype=float)
    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)
    close = np.array(close, dtype=float)

    log_oc = np.log(close / open_)
    log_cc = np.log(close / np.roll(close, 1))
    log_co = np.log(open_ / np.roll(close, 1))
    log_oh = np.log(high / open_)
    log_ol = np.log(low / open_)
    log_cc[0] = np.nan
    log_co[0] = np.nan

    rs = log_oh * (log_oh - log_oc) + log_ol * (log_ol - log_oc)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    yz_vol = np.full(n, np.nan)
    avg_vol = calc_sma(volumes, window) if volumes is not None else None

    for i in range(window - 1, n):
        sl = slice(i - window + 1, i + 1)
        cc = log_cc[sl][~np.isnan(log_cc[sl])]
        co = log_co[sl][~np.isnan(log_co[sl])]
        oc = log_oc[sl][~np.isnan(log_oc[sl])]
        r = rs[sl][~np.isnan(rs[sl])]
        if len(cc) < 2:
            continue

        yz_var = np.var(co, ddof=1) + k * np.var(oc, ddof=1) + (1 - k) * np.mean(r)
        base_vol = np.sqrt(max(yz_var, 0)) * np.sqrt(252) * 100.0

        price_move = close[i] - open_[i]
        candle_range = high[i] - low[i]
        flow_pressure = (price_move / candle_range) if candle_range > 0 else 0.0
        if volumes is not None and avg_vol is not None and not np.isnan(avg_vol[i]) and avg_vol[i] > 0:
            flow_pressure *= min(volumes[i] / avg_vol[i], 3.0)

        yz_vol[i] = base_vol * (1.0 + 0.3 * np.tanh(flow_pressure))

    return yz_vol


def compute_volatility(df, method, window=20):
    close = df['Close'].values
    open_ = df['Open'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    log_ret = calc_log_returns(close)

    if method in ("GARCH", "GARCH % Change"):
        raw = calc_volume_weighted_garch(log_ret, volume, window=window)
        label = "GARCH Volatility (%)"
    else:
        raw = calc_yang_zhang_vol(open_, high, low, close, volume, window=window)
        label = "Yang-Zhang Volatility (%)"

    if method in ("GARCH % Change", "YZ % Change"):
        raw = calc_pct_change(raw)
        label = label.replace("(%)", "% Chg")

    return pd.Series(raw, index=df.index, name=label)