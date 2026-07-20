"""
Cost-aware backtest metrics, including the Deflated Sharpe Ratio.

Save this file as: src/backtest.py

Formula source: Bailey, D. and Lopez de Prado, M. (2014), "The Deflated
Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and
Non-Normality," Journal of Portfolio Management, 40(5).

IMPORTANT convention note: this implementation uses PEARSON kurtosis
(normal distribution = 3), matching the original paper's notation --
NOT scipy's default EXCESS kurtosis (normal = 0). We call
scipy.stats.kurtosis(..., fisher=False) explicitly to get this right.
Silently mixing the two conventions is a common, hard-to-notice source
of a wrong DSR -- if you ever modify this, keep that call as-is.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return (returns.mean() / downside.std()) * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return drawdown.min()


def win_rate(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return (returns > 0).mean()


def probabilistic_sharpe_ratio(
    sr_hat: float, sr_benchmark: float, n_obs: int, skewness: float, pearson_kurt: float
) -> float:
    """
    PSR: probability that the TRUE Sharpe ratio exceeds sr_benchmark,
    given an estimated Sharpe sr_hat computed over n_obs observations.
    """
    numerator = (sr_hat - sr_benchmark) * np.sqrt(n_obs - 1)
    denominator = np.sqrt(1 - skewness * sr_hat + ((pearson_kurt - 1) / 4) * sr_hat ** 2)
    if denominator == 0 or np.isnan(denominator):
        return np.nan
    return norm.cdf(numerator / denominator)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """
    Expected maximum Sharpe ratio across n_trials independent trials,
    under the null that every trial's TRUE Sharpe is zero. This is the
    benchmark DSR tests your actual result against -- the more configs
    you tried in Phase 8-9, the higher this bar climbs, exactly as it
    should.
    """
    euler_mascheroni = 0.5772156649
    sr_std = np.sqrt(sr_variance)
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    return sr_std * ((1 - euler_mascheroni) * z1 + euler_mascheroni * z2)


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int, sr_variance: float = None) -> dict:
    """
    Full DSR pipeline: computes the estimated Sharpe from `returns`, the
    expected-max-Sharpe benchmark from `n_trials`, and the resulting
    probability that your strategy reflects genuine skill rather than
    being the best of N noisy trials.

    Parameters
    ----------
    returns : pd.Series
        Per-trade net returns, AFTER costs -- always feed this the
        `net_return_pct` output of costs.py's round_trip_cost(), never
        gross returns.
    n_trials : int
        Total number of distinct configurations tried during Phase 8-9
        (every feature set / barrier multiplier / hyperparameter
        combination). You must log this as you go -- it cannot be
        reconstructed after the fact.
    sr_variance : float, optional
        Variance of the Sharpe ratio across your logged trials. If you
        have every trial's Sharpe recorded, pass
        np.var(all_trial_sharpes) for an accurate DSR. If omitted, falls
        back to a rough 1/n_obs approximation -- logging real per-trial
        Sharpes is strongly preferred over relying on this fallback.

    Returns
    -------
    dict with sr_hat, sr_benchmark, dsr, n_obs, n_trials, skewness,
    pearson_kurtosis.
    """
    n_obs = len(returns)
    sr_hat = sharpe_ratio(returns, periods_per_year=1)  # per-trade, not annualized, for this formula
    skewness = skew(returns)
    pearson_kurt = kurtosis(returns, fisher=False)

    if sr_variance is None:
        sr_variance = 1.0 / n_obs

    sr_benchmark = expected_max_sharpe(n_trials, sr_variance)
    dsr = probabilistic_sharpe_ratio(sr_hat, sr_benchmark, n_obs, skewness, pearson_kurt)

    return {
        "sr_hat": sr_hat,
        "sr_benchmark": sr_benchmark,
        "dsr": dsr,
        "n_obs": n_obs,
        "n_trials": n_trials,
        "skewness": skewness,
        "pearson_kurtosis": pearson_kurt,
    }


def backtest_summary(returns: pd.Series, benchmark_returns: pd.Series = None) -> dict:
    """
    Standard backtest metric bundle. Always pass benchmark_returns (e.g.
    NIFTY buy-and-hold over the identical period) -- a positive Sharpe
    means little on its own if the benchmark did better, risk-adjusted,
    over the same stretch.
    """
    summary = {
        "total_return": (1 + returns).prod() - 1,
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "win_rate": win_rate(returns),
        "n_trades": len(returns),
    }

    if benchmark_returns is not None:
        summary["benchmark_total_return"] = (1 + benchmark_returns).prod() - 1
        summary["benchmark_sharpe"] = sharpe_ratio(benchmark_returns)
        summary["excess_return"] = summary["total_return"] - summary["benchmark_total_return"]

    return summary


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    strategy_returns = pd.Series(rng.normal(0.001, 0.02, 250))
    benchmark_returns = pd.Series(rng.normal(0.0005, 0.015, 250))

    print("Backtest summary:")
    for k, v in backtest_summary(strategy_returns, benchmark_returns).items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")

    print("\nDeflated Sharpe Ratio, honestly accounting for 50 configs tried:")
    dsr_result = deflated_sharpe_ratio(strategy_returns, n_trials=50)
    for k, v in dsr_result.items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")

    print("\nSame data, but pretending only 1 config was tried (for comparison):")
    dsr_result_1trial = deflated_sharpe_ratio(strategy_returns, n_trials=1.0000001)
    for k, v in dsr_result_1trial.items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")