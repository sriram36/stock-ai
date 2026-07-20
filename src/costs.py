"""
Transaction cost modeling for Indian equity trades (2026 rates).

Save this file as: src/costs.py

Rates below reflect the 2026-27 structure discussed earlier in this build.
STT and other government-levied charges change with Union Budgets --
re-verify these against your broker's current published charges before
relying on this for real capital. Do not assume they stay fixed.
"""

from dataclasses import dataclass


@dataclass
class CostConfig:
    """
    Defaults model a typical discount broker (zero delivery brokerage,
    flat intraday fee) under 2026 rates. Override any field to match your
    actual broker's published charges once you've picked one (Fyers/Dhan).
    """
    stt_delivery_pct: float = 0.001         # 0.1%, both legs
    stt_intraday_sell_pct: float = 0.00025  # 0.025%, sell side only
    brokerage_delivery: float = 0.0         # flat fee per leg, most discount brokers
    brokerage_intraday: float = 20.0        # flat fee per executed order (typical)
    stamp_duty_buy_pct: float = 0.00003     # 0.003% buy side, delivery
    exchange_txn_pct: float = 0.0000297     # NSE exchange transaction charge, approx
    gst_pct: float = 0.18                   # GST on (brokerage + exchange charges)
    slippage_bps: float = 5.0               # conservative placeholder -- replace
                                             # with real fill data once you're live


def round_trip_cost(
    entry_price: float,
    exit_price: float,
    quantity: int,
    trade_type: str = "delivery",
    config: CostConfig = None,
) -> dict:
    """
    Compute the all-in cost of a round-trip trade (entry + exit).

    Returns a dict with the full cost breakdown and net return after
    costs -- deliberately itemized, not just a single deducted number, so
    you can see exactly what's eating your gross P&L (usually STT and
    slippage dominate, not brokerage, on delivery trades).

    Always feed net_return_pct (not gross) into backtest.py's
    backtest_summary() and deflated_sharpe_ratio() -- a backtest on gross
    returns is not a real backtest.
    """
    if config is None:
        config = CostConfig()
    if trade_type not in ("delivery", "intraday"):
        raise ValueError("trade_type must be 'delivery' or 'intraday'")

    buy_value = entry_price * quantity
    sell_value = exit_price * quantity
    gross_pnl = sell_value - buy_value

    if trade_type == "delivery":
        stt = config.stt_delivery_pct * (buy_value + sell_value)
        brokerage = config.brokerage_delivery * 2
    else:
        stt = config.stt_intraday_sell_pct * sell_value
        brokerage = config.brokerage_intraday * 2

    stamp_duty = config.stamp_duty_buy_pct * buy_value
    exchange_txn = config.exchange_txn_pct * (buy_value + sell_value)
    gst = config.gst_pct * (brokerage + exchange_txn)
    slippage = (config.slippage_bps / 10_000) * (buy_value + sell_value)

    total_cost = stt + brokerage + stamp_duty + exchange_txn + gst + slippage
    net_pnl = gross_pnl - total_cost
    net_return_pct = net_pnl / buy_value if buy_value else 0.0

    return {
        "gross_pnl": gross_pnl,
        "stt": stt,
        "brokerage": brokerage,
        "stamp_duty": stamp_duty,
        "exchange_txn": exchange_txn,
        "gst": gst,
        "slippage": slippage,
        "total_cost": total_cost,
        "net_pnl": net_pnl,
        "net_return_pct": net_return_pct,
    }


if __name__ == "__main__":
    result = round_trip_cost(
        entry_price=500.0, exit_price=520.0, quantity=100, trade_type="delivery"
    )
    for k, v in result.items():
        print(f"{k:20s}: {v:,.2f}")