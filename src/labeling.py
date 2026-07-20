"""
Triple-Barrier labeling (Lopez de Prado, "Advances in Financial Machine Learning").

Save this file as: src/labeling.py

Design decisions (read before using):

1. Barriers are sized as multiples of ROLLING VOLATILITY, not a fixed percentage.
   This makes labels comparable across different stocks and different volatility
   regimes of the same stock -- a 2% move means something different in a calm
   market vs. an earnings-week spike.

2. Barrier touches are checked against intraday HIGH/LOW, not just Close.
   A stop-loss or take-profit order fires the moment price crosses it intraday --
   using only the daily close would miss touches that happened mid-day and
   reversed by close, which silently makes the backtest look better than reality.

3. Tie-break rule when both barriers are touched on the SAME day (can't be
   resolved from daily OHLC data alone): the stop-loss wins. This is the
   conservative assumption -- it never overstates profitability -- and it's
   the standard convention used in the reference literature for this exact
   ambiguity.

4. The output includes a t1 column (the date each label's outcome was
   resolved). This is not optional bookkeeping -- Phase 8's CombinatorialPurgedCV
   needs [t0, t1] for every label to know which training samples overlap with
   a given test window and must be purged. Keep this column all the way through
   the pipeline.
"""

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"open", "high", "low", "close"}


def daily_volatility(close: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Rolling standard deviation of daily returns, used to size barriers
    dynamically per entry point.

    Parameters
    ----------
    close : pd.Series
        Close prices, DatetimeIndex, sorted ascending.
    lookback : int
        Rolling window (trading days) for the volatility estimate.

    Returns
    -------
    pd.Series of the same index as `close`. First `lookback` values are NaN.
    """
    returns = close.pct_change()
    return returns.rolling(lookback).std()


def _validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df must be sorted ascending by date")


def triple_barrier_labels(
    df: pd.DataFrame,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    vertical_days: int = 10,
    vol_lookback: int = 20,
    min_vol: float = 1e-6,
) -> pd.DataFrame:
    """
    Generate triple-barrier labels for every valid entry date in `df`.

    Parameters
    ----------
    df : pd.DataFrame
        Must have lowercase columns: open, high, low, close.
        DatetimeIndex, sorted ascending, one row per trading day.
    pt_mult : float
        Profit-take barrier, as a multiple of daily volatility.
    sl_mult : float
        Stop-loss barrier, as a multiple of daily volatility.
    vertical_days : int
        Max holding period (trading days) before the vertical (time) barrier
        closes the position regardless of price.
    vol_lookback : int
        Window for the rolling volatility estimate that sizes the barriers.
    min_vol : float
        Entry dates with volatility below this are skipped (avoids
        degenerate barriers on near-zero-volatility days, e.g. circuit-halt
        artifacts).

    Returns
    -------
    pd.DataFrame indexed by t0 (entry date) with columns:
        t1          : date the position closed (barrier touch or vertical)
        entry_price : close price at t0
        exit_price  : price at t1 (barrier level if pt/sl, close if vertical)
        ret          : (exit_price / entry_price) - 1
        label        : +1 profit-take hit, -1 stop-loss hit, 0 timed out
        barrier      : 'pt', 'sl', or 'vertical'
        vol          : volatility used to size that entry's barriers
    """
    _validate_input(df)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = daily_volatility(close, lookback=vol_lookback)

    n = len(df)
    rows = []

    for loc0 in range(n):
        t0 = df.index[loc0]
        v = vol.iloc[loc0]

        if pd.isna(v) or v < min_vol:
            continue  # not enough history yet, or degenerate volatility

        max_loc = min(loc0 + vertical_days, n - 1)
        if max_loc <= loc0:
            continue  # not enough future data left to hold the position

        entry_price = close.iloc[loc0]
        upper = entry_price * (1 + pt_mult * v)
        lower = entry_price * (1 - sl_mult * v)

        path_high = high.iloc[loc0 + 1 : max_loc + 1]
        path_low = low.iloc[loc0 + 1 : max_loc + 1]

        upper_hits = path_high.index[path_high >= upper]
        lower_hits = path_low.index[path_low <= lower]

        first_upper = upper_hits[0] if len(upper_hits) else None
        first_lower = lower_hits[0] if len(lower_hits) else None

        if first_upper is None and first_lower is None:
            # neither barrier touched -> vertical barrier closes it
            t1 = df.index[max_loc]
            exit_price = close.iloc[max_loc]
            label, barrier = 0, "vertical"

        elif first_upper is not None and (
            first_lower is None or first_upper < first_lower
        ):
            t1, exit_price, label, barrier = first_upper, upper, 1, "pt"

        elif first_lower is not None and (
            first_upper is None or first_lower < first_upper
        ):
            t1, exit_price, label, barrier = first_lower, lower, -1, "sl"

        else:
            # same-day tie: stop-loss wins (conservative convention, see module docstring)
            t1, exit_price, label, barrier = first_lower, lower, -1, "sl"

        ret = (exit_price / entry_price) - 1

        rows.append(
            {
                "t0": t0,
                "t1": t1,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "ret": ret,
                "label": label,
                "barrier": barrier,
                "vol": v,
            }
        )

    return pd.DataFrame(rows).set_index("t0")


if __name__ == "__main__":
    # Quick self-check on synthetic data -- not a substitute for testing on
    # real data, just proves the function runs and the barrier logic is sane.
    rng = np.random.default_rng(42)
    n_days = 300
    dates = pd.bdate_range("2023-01-01", periods=n_days)

    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n_days)))
    daily_range = np.abs(rng.normal(0, 0.008, n_days)) * price

    synthetic = pd.DataFrame(
        {
            "open": price,
            "high": price + daily_range,
            "low": price - daily_range,
            "close": price,
        },
        index=dates,
    )

    labels = triple_barrier_labels(
        synthetic, pt_mult=2.0, sl_mult=2.0, vertical_days=10, vol_lookback=20
    )

    print(f"Generated {len(labels)} labels from {n_days} days of synthetic data")
    print(labels["label"].value_counts())
    print("\nSample rows:")
    print(labels.head())