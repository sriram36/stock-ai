"""
Main research driver: multi-stock cross-sectional pipeline.

Save this file as: src/run_research.py

Ties together every module built so far: data_collection -> features ->
labeling -> cross-sectional dataset assembly -> CPCV + Optuna
hyperparameter search -> final chronological holdout backtest -> cost-aware
returns -> deflated Sharpe Ratio.

This is the LOCAL/COLAB RESEARCH SCRIPT (Phases 2-11 of the build plan) --
it is NOT the daily automation job (Phase 14) or live execution (Phase 15).
Run this to answer exactly one question: does this strategy have any real,
cost-adjusted, honestly-deflated edge? Only proceed to automation/live
infrastructure if the answer here is yes.

Run modes
---------
`python run_research.py`                  -> smoke test on synthetic data
`python run_research.py --real --n=30`    -> real NSE data, N stocks from
                                              STOCK_UNIVERSE, via jugaad-data
"""

import argparse
from datetime import date

import numpy as np
import pandas as pd
import optuna

from features import build_features
from labeling import triple_barrier_labels
from validation import build_cpcv
from models import train_lightgbm
from costs import round_trip_cost
from backtest import backtest_summary, deflated_sharpe_ratio


# 30 liquid NIFTY 50 constituents, diversified across 11 sectors, all with
# long listing history (verified against the current NIFTY 50 list, July
# 2026) for adequate regime coverage in CPCV. Newer/recently-demerged
# constituents (Eternal, IndiGo, Max Healthcare, Jio Financial, Tata Motors
# Passenger Vehicles) were deliberately excluded -- not enough history yet
# for a meaningful multi-year backtest.
STOCK_UNIVERSE = {
    "RELIANCE": "Oil_Gas", "ONGC": "Oil_Gas",
    "HDFCBANK": "Financials", "ICICIBANK": "Financials", "SBIN": "Financials",
    "KOTAKBANK": "Financials", "AXISBANK": "Financials", "BAJFINANCE": "Financials",
    "INFY": "IT", "TCS": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "TATACONSUM": "FMCG",
    "BHARTIARTL": "Telecom",
    "LT": "Construction",
    "MARUTI": "Auto", "M&M": "Auto", "BAJAJ-AUTO": "Auto", "EICHERMOT": "Auto",
    "SUNPHARMA": "Healthcare", "CIPLA": "Healthcare", "DRREDDY": "Healthcare",
    "TATASTEEL": "Metals", "HINDALCO": "Metals", "JSWSTEEL": "Metals",
    "GRASIM": "Cement", "ULTRACEMCO": "Cement",
    "NTPC": "Power",
}
assert len(STOCK_UNIVERSE) == 30

# NOTE: "M&M" and "BAJAJ-AUTO" contain non-alphanumeric characters (&, -).
# These are the real, current NSE symbols -- but verify jugaad-data's
# stock_df() handles them cleanly before relying on this in an automated
# run; if either fails, drop it from the dict and re-run (29 stocks is
# still a perfectly fine cross-sectional universe).


def fetch_universe(symbols, start: date, end: date) -> dict:
    """
    Fetch real historical data for a list of symbols via jugaad-data.
    Requires network access to nseindia.com -- run locally, not in a
    sandboxed environment without that access.

    Prints elapsed time per symbol -- fetching 30 stocks x ~10 years each
    legitimately takes several minutes (NSE rate limits + per-symbol
    parsing). This is normal, not a hang -- don't Ctrl+C early.
    """
    import time as _time
    from data_collection import fetch_nse_history

    stock_data = {}
    run_start = _time.time()

    for i, symbol in enumerate(symbols, 1):
        t0 = _time.time()
        try:
            print(f"  [{i}/{len(symbols)}] fetching {symbol}...", end=" ", flush=True)
            stock_data[symbol] = fetch_nse_history(symbol, start, end)
            print(f"done in {_time.time() - t0:.1f}s ({len(stock_data[symbol])} rows)")
        except Exception as e:
            print(f"FAILED after {_time.time() - t0:.1f}s ({e}) -- skipping, continuing with the rest")

    print(f"\n  Total fetch time: {(_time.time() - run_start) / 60:.1f} minutes, "
          f"{len(stock_data)}/{len(symbols)} stocks succeeded")
    return stock_data


def build_cross_sectional_dataset(
    stock_data: dict, universe: dict, pt_mult=2.0, sl_mult=2.0, vertical_days=10
):
    """
    Build one combined, cross-sectional feature+label dataset from
    multiple stocks -- one model learns from all of them at once
    (Phase 18's recommended approach over per-stock models).

    Returns
    -------
    X : features (incl. `sector`, `stock_id` as pandas 'category' dtype --
        LightGBM handles these natively without one-hot encoding)
    y : 1 = profit-take hit, 0 = stop-loss hit (timeouts dropped)
    t1 : label resolution date, kept for reference
    symbols : which stock each row belongs to, kept for the final backtest
    """
    rows = []
    for symbol, df in stock_data.items():
        sector = universe[symbol]
        feats = build_features(df)
        labels = triple_barrier_labels(
            df, pt_mult=pt_mult, sl_mult=sl_mult, vertical_days=vertical_days
        )

        joined = feats.join(labels[["t1", "label"]], how="inner").dropna()
        joined = joined[joined["label"] != 0].copy()
        joined["sector"] = sector
        joined["stock_id"] = symbol
        joined["symbol"] = symbol
        rows.append(joined)

    # CRITICAL: sort by date across ALL stocks together, not stock-by-stock.
    # CPCV needs one unified chronological ordering to purge/embargo correctly.
    combined = pd.concat(rows).sort_index()

    combined["sector"] = combined["sector"].astype("category")
    combined["stock_id"] = combined["stock_id"].astype("category")

    y = (combined["label"] == 1).astype(int)
    t1 = combined["t1"]
    symbols = combined["symbol"]
    X = combined.drop(columns=["label", "t1", "symbol"])

    return X, y, t1, symbols


def build_cross_sectional_cpcv(vertical_days, n_stocks, n_folds=10, n_test_folds=8):
    """
    Multi-stock CPCV needs a WIDER purge/embargo window than the
    single-stock case in validation.py. Pooling N stocks and sorting by
    date means each calendar day can contribute up to N rows to the
    combined index -- so purging `vertical_days` ROWS around a fold
    boundary only covers a fraction of a calendar day, not
    `vertical_days` days, and silently under-purges (a real leakage risk).

    Conservative fix: scale purged_size/embargo_size by n_stocks. This
    over-purges on days when not every stock traded (holidays, listing
    gaps, data gaps) -- the safe direction to be wrong in.
    """
    return build_cpcv(
        vertical_days=vertical_days * n_stocks, n_folds=n_folds, n_test_folds=n_test_folds
    )


def run_optuna_search(X, y, cv, n_trials=30):
    """
    Wraps Optuna around the CPCV loop for hyperparameter search.

    IMPORTANT: n_trials here is the exact number that must be fed into
    deflated_sharpe_ratio() afterward (see __main__ below) -- this is
    the "how many configs did you try" count that Phase 11's overfitting
    correction depends on. Don't lose track of it, and don't run
    additional ad-hoc trials outside this function without adding them
    to the count.
    """
    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        }

        fold_scores = []
        for train_idx, test_idx_groups in cv.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            test_idx = np.concatenate(test_idx_groups)
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue  # degenerate fold, skip

            model = train_lightgbm(X_train, y_train, X_test, y_test, params=params)
            preds = model.predict(X_test)
            # Simple accuracy as the search objective for now. Once you have
            # a working baseline, swap this for a cost-aware Sharpe proxy --
            # accuracy and profitability are not the same thing (see the
            # very first message in this build's history).
            score = ((preds > 0.5).astype(int) == y_test).mean()
            fold_scores.append(score)

        return float(np.mean(fold_scores)) if fold_scores else 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def final_backtest(X, y, symbols, labels_full, best_params, n_trials, test_frac=0.2):
    """
    After CPCV+Optuna model SELECTION, run ONE final chronological
    holdout backtest as the honest "would this have worked" evaluation.
    The CPCV folds were used to pick hyperparameters -- they cannot also
    be your reported performance without a second layer of optimism.

    Uses labels_full (the original triple-barrier output, with real
    entry/exit prices) to compute genuine cost-aware returns via
    costs.py -- not just classification accuracy, which is not the
    same thing as profitability.
    """
    unique_dates = X.index.unique().sort_values()
    cutoff = unique_dates[int(len(unique_dates) * (1 - test_frac))]

    train_mask = X.index < cutoff
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if y_train.nunique() < 2 or y_test.nunique() < 2 or len(X_test) == 0:
        print("   Not enough data/class diversity for a final holdout -- skipping.")
        return None

    model = train_lightgbm(X_train, y_train, X_test, y_test, params=best_params)
    preds = model.predict(X_test)
    predicted_long = preds > 0.5

    test_symbols = symbols[test_mask]

    net_returns = []
    for idx, sym, went_long in zip(X_test.index, test_symbols, predicted_long):
        if not went_long:
            # Model predicted a down move -- skip the trade entirely in this
            # simple long-only strategy rather than shorting it.
            continue
        row = labels_full.loc[(labels_full.index == idx) & (labels_full["__symbol__"] == sym)]
        if row.empty:
            continue
        row = row.iloc[0]
        cost_result = round_trip_cost(row["entry_price"], row["exit_price"], quantity=10, trade_type="delivery")
        net_returns.append(cost_result["net_return_pct"])

    if not net_returns:
        print("   Model predicted no long trades on the holdout -- nothing to backtest.")
        return None

    net_returns = pd.Series(net_returns)
    summary = backtest_summary(net_returns)
    dsr = deflated_sharpe_ratio(net_returns, n_trials=n_trials)

    return summary, dsr


def _synthetic_smoke_test_data(symbols, n_days=400, seed=42):
    """Synthetic OHLCV for proving the pipeline runs end to end without real data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    stock_data = {}
    for symbol in symbols:
        price = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n_days)))
        rng_range = np.abs(rng.normal(0, 0.008, n_days)) * price
        stock_data[symbol] = pd.DataFrame(
            {
                "open": price, "high": price + rng_range, "low": price - rng_range,
                "close": price, "volume": rng.integers(100000, 1000000, n_days),
            },
            index=dates,
        )
    return stock_data


def main(real: bool, n_stocks: int, n_trials: int):
    symbols = list(STOCK_UNIVERSE.keys())[:n_stocks]

    if real:
        print(f"Fetching real NSE data for {len(symbols)} stocks (this can take a few minutes)...")
        stock_data = fetch_universe(symbols, start=date(2015, 1, 1), end=date.today())
    else:
        print(f"SMOKE TEST MODE -- synthetic data for {len(symbols)} stocks (pass --real for actual NSE data)")
        stock_data = _synthetic_smoke_test_data(symbols)

    if len(stock_data) < 2:
        print("Fewer than 2 stocks fetched successfully -- aborting.")
        return

    print(f"\n1. Building cross-sectional dataset from {len(stock_data)} stocks...")
    X, y, t1, symbols_col = build_cross_sectional_dataset(stock_data, STOCK_UNIVERSE)
    print(f"   {len(X)} total rows, {X.shape[1]} features (incl. sector/stock_id)")
    print(f"   class balance: {dict(y.value_counts())}")

    labels_parts = []
    for symbol, df in stock_data.items():
        lbl = triple_barrier_labels(df, pt_mult=2.0, sl_mult=2.0, vertical_days=10)
        lbl["__symbol__"] = symbol
        labels_parts.append(lbl)
    labels_full = pd.concat(labels_parts)

    print("2. Building cross-sectional CPCV (purge/embargo scaled for pooled stocks)...")
    cv = build_cross_sectional_cpcv(vertical_days=10, n_stocks=len(stock_data), n_folds=6, n_test_folds=2)

    print(f"3. Running Optuna search ({n_trials} trials)...")
    study = run_optuna_search(X, y, cv, n_trials=n_trials)
    print(f"   best CPCV score: {study.best_value:.4f}")
    print(f"   best params: {study.best_params}")

    print("4. Final chronological holdout backtest (cost-aware, after real STT/brokerage/slippage)...")
    result = final_backtest(X, y, symbols_col, labels_full, study.best_params, n_trials=n_trials)
    if result:
        summary, dsr = result
        print("\n   Backtest summary:")
        for k, v in summary.items():
            print(f"     {k:20s}: {v:.4f}" if isinstance(v, float) else f"     {k:20s}: {v}")
        print("\n   Deflated Sharpe Ratio (the real headline number, honestly accounting for"
              f" {n_trials} configs tried):")
        for k, v in dsr.items():
            print(f"     {k:20s}: {v:.4f}" if isinstance(v, float) else f"     {k:20s}: {v}")

        print(
            "\n   Interpretation: DSR is the probability your strategy has genuine skill,"
            " not luck. Above ~0.95 is the usual bar to take seriously. Below that, the"
            " honest conclusion is 'no demonstrated edge yet' -- a real, valid, and common"
            " outcome at this stage. See Phase 11 of the build plan."
        )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use real NSE data instead of synthetic smoke test")
    parser.add_argument("--n", type=int, default=5, help="Number of stocks from STOCK_UNIVERSE to use")
    parser.add_argument("--trials", type=int, default=5, help="Number of Optuna trials")
    args = parser.parse_args()

    main(real=args.real, n_stocks=args.n, n_trials=args.trials)