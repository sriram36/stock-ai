"""
Feature engineering: technical indicators computed directly with pandas/numpy.

Save this file as: src/features.py

Deliberately hand-rolled instead of using pandas-ta: the original pandas-ta
package carries an explicit maintainer warning as of 2026 ("current levels
are unsustainable and risks discontinuation"), and the community fork
(pandas-ta-classic) is newer and less battle-tested. For the small, well-known
set of indicators used here, a transparent from-scratch implementation is
more reliable than depending on either -- and it's fully auditable, which
matters when real money rides on these numbers being correct.
"""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def _validate(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {missing}")


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range -- also used by labeling.py to size triple-barrier widths."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window).mean()


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(close, window)
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (close - lower) / (upper - lower)
    return upper, mid, lower, pct_b


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    atr_ = tr.rolling(window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(window).mean()


def build_features(
    df: pd.DataFrame, lag_periods=(1, 3, 5, 10), vol_lookback: int = 20
) -> pd.DataFrame:
    """
    Main entry point. df must have lowercase columns: open, high, low,
    close, volume -- same schema fetch_nse_history() in data_collection.py
    returns.

    Returns a DataFrame of the same index as df, with one column per
    feature. Early rows will be NaN until each indicator's lookback window
    is satisfied -- drop these before feeding into model training
    (models.py's prepare_dataset() already does this via dropna()).
    """
    _validate(df)
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    out["sma_10"] = sma(close, 10)
    out["sma_50"] = sma(close, 50)
    out["ema_12"] = ema(close, 12)
    out["ema_26"] = ema(close, 26)
    out["rsi_14"] = rsi(close, 14)

    macd_line, signal_line, hist = macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    out["atr_14"] = atr(df, 14)

    upper, mid, lower, pct_b = bollinger_bands(close)
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["bb_pct_b"] = pct_b

    out["adx_14"] = adx(df, 14)

    for lag in lag_periods:
        out[f"return_{lag}d"] = close.pct_change(lag)

    out["rolling_vol"] = close.pct_change().rolling(vol_lookback).std()
    out["volume_zscore"] = (
        (df["volume"] - df["volume"].rolling(vol_lookback).mean())
        / df["volume"].rolling(vol_lookback).std()
    )

    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.bdate_range("2023-01-01", periods=n)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    rng_range = np.abs(rng.normal(0, 0.008, n)) * price
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + rng_range,
            "low": price - rng_range,
            "close": price,
            "volume": rng.integers(100000, 1000000, n),
        },
        index=dates,
    )

    feats = build_features(df)
    print(feats.tail())
    print(f"\n{feats.shape[1]} features generated, {feats.dropna().shape[0]} complete rows")