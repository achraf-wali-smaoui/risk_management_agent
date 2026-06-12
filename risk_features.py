import pandas as pd
import numpy as np

from typing import Literal

TrendLabel = Literal["up", "down", "flat"]
LevelLabel = Literal["low", "mid", "high"]


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """Compute True Range (TR) from OHLC."""
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute ATR using Wilder-like EMA smoothing (common in finance).
    """
    tr = compute_true_range(df)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def safe_zscore(x: pd.Series, window: int) -> pd.Series:
    """Rolling z-score with safe handling for near-zero std."""
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return (x - mu) / sd



def _rolling_quantile_bounds(x: pd.Series, window: int, q_lo: float, q_hi: float) -> pd.DataFrame:
    """Compute rolling quantile bounds for volatility levels."""
    ql = x.rolling(window).quantile(q_lo)
    qh = x.rolling(window).quantile(q_hi)
    return pd.DataFrame({"q_lo": ql, "q_hi": qh})


def classify_level(vol: float, q_lo: float, q_hi: float) -> LevelLabel:
    """Classify volatility level based on quantile bounds."""
    if np.isnan(vol) or np.isnan(q_lo) or np.isnan(q_hi):
        return "mid"
    if vol < q_lo:
        return "low"
    if vol > q_hi:
        return "high"
    return "mid"

def classify_trend(vol_series: pd.Series, trend_fraction: float) -> TrendLabel:
    """
    Classify volatility trend by comparing short-term mean vs long-term mean.

    short length = max(2, floor(rho * W))
    long mean = mean over full W
    """
    if vol_series.isna().all():
        return "flat"
    W = len(vol_series)
    if W < 4:
        return "flat"

    m = max(2, int(np.floor(trend_fraction * W)))
    long_mean = float(np.nanmean(vol_series.values))
    short_mean = float(np.nanmean(vol_series.values[-m:]))

    eps = 1e-6
    if short_mean > long_mean + eps:
        return "up"
    if short_mean < long_mean - eps:
        return "down"
    return "flat"


def build_qr_features(vol: pd.Series, window: int) -> pd.DataFrame:
    """
    Build interpretable features for quantile regression using ONLY past volatility values.
    """
    df = pd.DataFrame({"vol": vol})

    # Lags
    for k in (1, 2, 3):
        df[f"lag_{k}"] = df["vol"].shift(k)

    # Rolling stats
    df["roll_mean"] = df["vol"].rolling(window).mean()
    df["roll_std"] = df["vol"].rolling(window).std(ddof=0)
    df["roll_min"] = df["vol"].rolling(window).min()
    df["roll_max"] = df["vol"].rolling(window).max()

    # Simple trend proxy ratio
    short_w = max(2, int(np.floor(0.2 * window)))
    df["short_mean"] = df["vol"].rolling(short_w).mean()
    df["trend_ratio"] = df["short_mean"] / (df["roll_mean"].replace(0.0, np.nan))

    df = df.drop(columns=["vol", "short_mean"])
    return df
