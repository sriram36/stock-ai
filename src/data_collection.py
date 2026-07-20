"""
Historical OHLCV collection from NSE via jugaad-data.

Save this file as: src/data_collection.py

jugaad-data scrapes the NSE website -- it is NOT an official API and can
break if NSE changes their site. This is intentional and safe by design:
it is used ONLY for research/backtesting data (Phase 2 of the plan),
never in the live execution path (Phase 15 uses the Fyers API instead).
If this breaks, your research stalls -- your capital doesn't.

Windows note: jugaad-data's stock_df() internally uses a ThreadPoolExecutor
to fetch date-range chunks in parallel, and creates a local cache directory
on first use. On Windows, the very first call in a fresh environment can
race on creating that cache directory and raise
`OSError: [WinError 183] Cannot create a file when that file already
exists`. It's a one-time issue -- once the directory exists, subsequent
calls (even for other symbols) don't hit it. fetch_nse_history() below
retries automatically on this specific error class.

Install: pip install jugaad-data
"""

import time
import warnings
from datetime import date
import pandas as pd

try:
    from jugaad_data.nse import stock_df
except ImportError as e:
    raise ImportError("jugaad-data not installed. Run: pip install jugaad-data") from e

# Harmless internal warning from jugaad-data's datetime handling -- fires
# once per row on large date ranges and floods the console enough to make
# a working fetch look hung. Silenced here, not swept under the rug --
# it doesn't affect correctness of the OHLCV values themselves.
warnings.filterwarnings(
    "ignore", message="no explicit representation of timezones", category=UserWarning
)


# jugaad-data's stock_df() returns these uppercase columns (verified against
# current docs/examples) -- map to the lowercase schema used throughout the
# rest of this pipeline (labeling.py, features.py).
COLUMN_MAP = {
    "DATE": "date",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "VOLUME": "volume",
}


def _parse_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    The transform logic, kept separate from the network call for
    testability. If NSE changes their site format, this is what will
    raise a clear error rather than silently producing wrong data.
    """
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(
            f"jugaad-data returned unexpected columns, missing: {missing}. "
            f"Got: {list(df.columns)}. NSE may have changed their site format -- "
            f"check jugaad-data's GitHub issues before debugging further."
        )

    df = df[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["open", "high", "low", "close"])


def fetch_nse_history(
    symbol: str,
    start: date,
    end: date,
    series: str = "EQ",
    max_retries: int = 2,
    retry_wait: float = 3.0,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV history for one NSE symbol.

    Parameters
    ----------
    symbol : str
        NSE symbol, e.g. "SBIN", "RELIANCE", "TCS". No exchange suffix.
    start, end : date
        Inclusive date range.
    series : str
        NSE series code, "EQ" for standard equity delivery series.
    max_retries : int
        Retries on OSError (covers the Windows cache-directory race
        described in the module docstring). Other exception types
        (network errors, no-data errors) are not retried -- they need
        a human to look, not a blind retry.
    retry_wait : float
        Seconds to wait between retries.

    Returns
    -------
    pd.DataFrame indexed by date (DatetimeIndex, ascending), columns:
        open, high, low, close, volume
    This is the exact schema build_features() and triple_barrier_labels()
    expect -- no renaming needed downstream.

    Note: jugaad-data's stock_df() already internally chunks and
    parallelizes across the requested date range -- no manual chunking
    loop needed here, which also means fewer redundant cache-directory
    touches than an earlier version of this function had.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            raw = stock_df(symbol=symbol, from_date=start, to_date=end, series=series)
            if raw is None or len(raw) == 0:
                raise ValueError(f"No data returned for {symbol} between {start} and {end}")
            return _parse_raw(raw)
        except OSError as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(retry_wait)
                continue
            raise RuntimeError(
                f"{symbol}: failed after {max_retries + 1} attempts due to a repeated "
                f"OS-level error (last: {e}). This is usually a Windows cache-directory "
                f"issue -- try running again, it typically resolves after the first "
                f"successful fetch of any symbol."
            ) from e

    raise last_error


if __name__ == "__main__":
    # Requires network access to nseindia.com -- run this locally, not in a
    # sandboxed environment without that access.
    df = fetch_nse_history("SBIN", date(2023, 1, 1), date(2023, 6, 30))
    print(df.head())
    print(f"\n{len(df)} rows fetched")