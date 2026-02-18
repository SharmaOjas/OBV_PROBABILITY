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

    init_slice = log_returns[:min(window, n)]
    valid_init = init_slice[np.isfinite(init_slice)]
    if len(valid_init) == 0:
        return volatilities
    uncond_var = np.clip(np.mean(valid_init ** 2), 0.0001, 1.0)
    prev_variance = uncond_var
    prev_sq_return = uncond_var

    print(f"[GARCH] n={n}, window={window}")
    print(f"[GARCH] log_returns[:5]={log_returns[:5]}")
    print(f"[GARCH] valid_init[:5]={valid_init[:5]}")
    print(f"[GARCH] uncond_var={uncond_var:.8f}")
    print(f"[GARCH] prev_variance={prev_variance:.8f}, prev_sq_return={prev_sq_return:.8f}")

    for i in range(n):
        recent_vols = np.maximum(volumes[max(0, i - 199):i + 1], 0)
        avg_volume = np.nanmean(recent_vols)
        if not np.isfinite(avg_volume) or avg_volume <= 0:
            avg_volume = 1.0

        cur_vol = max(volumes[i], 0) if np.isfinite(volumes[i]) else 0.0
        vol_ratio = np.clip(cur_vol / avg_volume, 0.1, 10.0)

        cur_ret = log_returns[i] if np.isfinite(log_returns[i]) else 0.0
        sq_ret = cur_ret ** 2
        core_var = omega + alpha * prev_sq_return + beta * prev_variance
        vol_term = np.tanh(gamma * np.log(vol_ratio))
        new_var = core_var * (1.0 + vol_term)
        if not np.isfinite(new_var) or new_var < 0:
            new_var = prev_variance
        new_var = np.clip(new_var, 0.0001, 1.0)
        volatilities[i] = np.sqrt(new_var) * np.sqrt(252)

        if i < 5:
            print(f"[GARCH] i={i}")
            print(f"  avg_volume={avg_volume:.4f}, cur_vol={cur_vol:.4f}, vol_ratio={vol_ratio:.4f}")
            print(f"  cur_ret={cur_ret:.8f}, sq_ret={sq_ret:.8f}")
            print(f"  vol_term={vol_term:.6f}, core_var={core_var:.8f}")
            print(f"  new_var (pre-clip)={core_var * (1.0 + vol_term):.8f}")
            print(f"  new_var (post-clip)={new_var:.8f}")
            print(f"  volatility={volatilities[i]:.4f}")

        prev_variance = new_var
        prev_sq_return = sq_ret

    print(f"[GARCH] volatilities[:5]={volatilities[:5]}")
    print(f"[GARCH] volatilities[-5:]={volatilities[-5:]}")
    return volatilities

def calc_yang_zhang(open_, high, low, close, window=10):

    open_ = np.asarray(open_, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    n = len(close)
    result = np.full(n, np.nan)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    for i in range(window, n):

        vo = 0.0
        vc = 0.0
        vrs = 0.0

        # Rolling window
        for j in range(i - window + 1, i + 1):

            if j == 0:
                continue

            op = open_[j]
            hi = high[j]
            lo = low[j]
            cl = close[j]
            cl_prior = close[j - 1]

            # Overnight second moment
            no = np.log(op / cl_prior)
            vo += no * no

            # Open → Close second moment
            nc = np.log(cl / op)
            vc += nc * nc

            # Conditional Rogers-Satchell (JS filtered)
            if hi > max(op, cl):
                vrs += np.log(hi / cl) * np.log(hi / op)

            if lo < min(op, cl):
                vrs += np.log(lo / cl) * np.log(lo / op)

        variance = (
            vo / window
            + k * (vc / window)
            + (1 - k) * (vrs / window)
        )

        result[i] = np.sqrt(max(variance, 0.0))

    return result


# -------------------------------------------------
# VolFlow: Yang-Zhang × Order Flow Pressure
# -------------------------------------------------
def calc_yang_zhang_vol(
    open_,
    high,
    low,
    close,
    volumes=None,
    window=10,
):

    open_ = np.asarray(open_, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    n = len(close)

    yz_vol = calc_yang_zhang(
        open_, high, low, close, window
    )

    result = np.full(n, np.nan)

    avg_vol = calc_sma(volumes, window) if volumes is not None else None

    for i in range(window - 1, n):

        base_vol = yz_vol[i]
        if not np.isfinite(base_vol):
            continue

        # Flow pressure
        price_move = close[i] - open_[i]
        candle_range = high[i] - low[i]

        flow_pressure = 0.0
        if candle_range > 0:
            flow_pressure = price_move / candle_range

        # Volume confirmation
        if (
            volumes is not None
            and avg_vol is not None
            and np.isfinite(avg_vol[i])
            and avg_vol[i] > 0
        ):
            vol_ratio = min(volumes[i] / avg_vol[i], 3.0)
            flow_pressure *= vol_ratio

        # Nonlinear scaling
        flow_multiplier = 1.0 + 0.3 * np.tanh(flow_pressure)

        result[i] = base_vol * flow_multiplier * 100

    return result

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