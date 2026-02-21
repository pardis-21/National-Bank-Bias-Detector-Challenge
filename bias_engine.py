"""
bias_engine.py
--------------
Rule-based trading bias detection using Pandas.
All three detectors return a consistent dict:
    {
        "flagged": bool,
        "reasons": list[str],
        "details": dict,
        "flagged_trades": list[dict]   # only for revenge trading
    }

Expected DataFrame columns (case-sensitive):
    timestamp, buy_sell, asset, quantity,
    entry_price, exit_price, profit_loss, balance
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# CONSTANTS / DEFAULT THRESHOLDS
# ─────────────────────────────────────────────────────────────
DEFAULT_MAX_TRADES_PER_HOUR   = 10     # flag if single hour exceeds this
DEFAULT_VOL_BALANCE_RATIO     = 3.0    # flag if total_volume / avg_balance > this
DEFAULT_SWITCH_WINDOW_MIN     = 30     # minutes; rapid position-switch window
DEFAULT_MIN_SWITCHES          = 3      # minimum switches to flag

DEFAULT_LOSS_WIN_RATIO        = 1.5    # flag if avg_loss / avg_win > this
DEFAULT_PRICE_RANGE_RATIO     = 1.5    # flag if loss price-range / win price-range > this

DEFAULT_REVENGE_QTY_MULT      = 1.5    # flag if qty after loss > avg_qty * this
DEFAULT_REVENGE_TIME_MIN      = 15     # minutes; window after a loss to look for revenge trade


# ─────────────────────────────────────────────────────────────
# 1. OVERTRADING DETECTION
# ─────────────────────────────────────────────────────────────
def detect_overtrading(
    df: pd.DataFrame,
    max_per_hour: int   = DEFAULT_MAX_TRADES_PER_HOUR,
    max_vol_ratio: float = DEFAULT_VOL_BALANCE_RATIO,
    switch_window_min: int = DEFAULT_SWITCH_WINDOW_MIN,
    min_switches: int   = DEFAULT_MIN_SWITCHES,
) -> dict:
    """
    Detects overtrading via three sub-checks:
      A) Time-based clustering  — too many trades in one hour
      B) Volume-to-balance ratio — total notional far exceeds account size
      C) Rapid position switching — alternating buy/sell same asset in short window
    """
    result = {"flagged": False, "reasons": [], "details": {}, "flagged_trades": []}

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── A: Time-based clustering ──────────────────────────────
    hourly = df.set_index("timestamp").resample("1h").size()
    peak_val  = int(hourly.max())
    peak_hour = hourly.idxmax()

    result["details"]["peak_trades_in_hour"] = peak_val
    result["details"]["peak_hour"] = str(peak_hour)

    if peak_val > max_per_hour:
        result["flagged"] = True
        result["reasons"].append(
            f"Executed {peak_val} trades in one hour (threshold: {max_per_hour}) "
            f"at {peak_hour.strftime('%Y-%m-%d %H:%M')}."
        )

    # ── B: Volume-to-balance ratio ────────────────────────────
    if "entry_price" in df.columns and "quantity" in df.columns:
        df["trade_value"] = df["quantity"] * df["entry_price"]
        total_vol   = df["trade_value"].sum()
        avg_balance = df["balance"].mean() if "balance" in df.columns and df["balance"].mean() != 0 else 1
        ratio = round(total_vol / avg_balance, 2)

        result["details"]["volume_balance_ratio"] = ratio

        if ratio > max_vol_ratio:
            result["flagged"] = True
            result["reasons"].append(
                f"Total trade volume (${total_vol:,.0f}) is {ratio}x your average balance "
                f"(threshold: {max_vol_ratio}x)."
            )

    # ── C: Rapid position switching ───────────────────────────
    if "asset" in df.columns and "buy_sell" in df.columns:
        switches = 0
        for asset in df["asset"].unique():
            sub = df[df["asset"] == asset].reset_index(drop=True)
            for i in range(1, len(sub)):
                diff_min = (sub.loc[i, "timestamp"] - sub.loc[i-1, "timestamp"]).total_seconds() / 60
                if sub.loc[i, "buy_sell"] != sub.loc[i-1, "buy_sell"] and diff_min < switch_window_min:
                    switches += 1

        result["details"]["rapid_position_switches"] = switches

        if switches >= min_switches:
            result["flagged"] = True
            result["reasons"].append(
                f"Detected {switches} rapid position switches (same asset, opposite side, "
                f"within {switch_window_min} min)."
            )

    return result


# ─────────────────────────────────────────────────────────────
# 2. LOSS AVERSION DETECTION
# ─────────────────────────────────────────────────────────────
def detect_loss_aversion(
    df: pd.DataFrame,
    ratio_threshold: float     = DEFAULT_LOSS_WIN_RATIO,
    price_range_ratio: float   = DEFAULT_PRICE_RANGE_RATIO,
) -> dict:
    """
    Detects loss aversion via two sub-checks:
      A) Win/Loss size ratio  — average loss significantly larger than average win
      B) Price range asymmetry — exits winners early, rides losers far
    """
    result = {"flagged": False, "reasons": [], "details": {}, "flagged_trades": []}

    winners = df[df["profit_loss"] > 0]["profit_loss"]
    losers  = df[df["profit_loss"] < 0]["profit_loss"].abs()

    if winners.empty or losers.empty:
        result["details"]["note"] = "Insufficient data — need both wins and losses."
        return result

    avg_win  = round(winners.mean(), 2)
    avg_loss = round(losers.mean(), 2)
    ratio    = round(avg_loss / avg_win, 2) if avg_win > 0 else 0

    result["details"]["avg_win"]        = avg_win
    result["details"]["avg_loss"]       = avg_loss
    result["details"]["loss_win_ratio"] = ratio
    result["details"]["total_winners"]  = int(len(winners))
    result["details"]["total_losers"]   = int(len(losers))

    # ── A: P/L size ratio ─────────────────────────────────────
    if ratio > ratio_threshold:
        result["flagged"] = True
        result["reasons"].append(
            f"Average loss (${avg_loss}) is {ratio}x your average win (${avg_win}) "
            f"(threshold: {ratio_threshold}x) — you may be holding losers too long."
        )

    # ── B: Price range asymmetry ──────────────────────────────
    if "entry_price" in df.columns and "exit_price" in df.columns:
        df2 = df.copy()
        df2["price_range"] = (df2["exit_price"] - df2["entry_price"]).abs()

        avg_range_win  = round(df2[df2["profit_loss"] > 0]["price_range"].mean(), 2)
        avg_range_loss = round(df2[df2["profit_loss"] < 0]["price_range"].mean(), 2)
        range_ratio    = round(avg_range_loss / avg_range_win, 2) if avg_range_win > 0 else 0

        result["details"]["avg_price_move_winners"] = avg_range_win
        result["details"]["avg_price_move_losers"]  = avg_range_loss
        result["details"]["price_range_ratio"]      = range_ratio

        if range_ratio > price_range_ratio:
            result["flagged"] = True
            result["reasons"].append(
                f"Losing trades travel {range_ratio}x more price distance before exit "
                f"({avg_range_loss} vs {avg_range_win} for winners) — closing winners too "
                f"early and riding losses."
            )

    return result


# ─────────────────────────────────────────────────────────────
# 3. REVENGE TRADING DETECTION
# ─────────────────────────────────────────────────────────────
def detect_revenge_trading(
    df: pd.DataFrame,
    qty_multiplier: float = DEFAULT_REVENGE_QTY_MULT,
    time_window_min: int  = DEFAULT_REVENGE_TIME_MIN,
) -> dict:
    """
    Detects revenge trading:
      After any losing trade, checks the NEXT trade for:
        • Quantity significantly above the trader's historical average  AND/OR
        • Opened within `time_window_min` minutes of the loss
    """
    result = {"flagged": False, "reasons": [], "details": {}, "flagged_trades": []}

    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    if "quantity" not in df.columns:
        result["details"]["note"] = "No 'quantity' column — cannot detect revenge trading."
        return result

    avg_qty = df["quantity"].mean()
    result["details"]["average_quantity"] = round(float(avg_qty), 4)

    # Use .shift() to align each row with the previous trade's P/L and timestamp
    df["prev_pl"]       = df["profit_loss"].shift(1)
    df["prev_ts"]       = df["timestamp"].shift(1)
    df["time_since_prev"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds() / 60

    # A trade is flagged as revenge if:
    #   • The preceding trade was a loss (prev_pl < 0)
    #   • AND current quantity > avg_qty * multiplier
    #   • AND it was opened within the time window
    mask = (
        (df["prev_pl"] < 0) &
        (df["quantity"] > avg_qty * qty_multiplier) &
        (df["time_since_prev"] <= time_window_min)
    )
    flagged_df = df[mask]

    for _, row in flagged_df.iterrows():
        result["flagged_trades"].append({
            "timestamp":      str(row["timestamp"]),
            "asset":          row.get("asset", "N/A"),
            "quantity":       round(float(row["quantity"]), 4),
            "avg_quantity":   round(float(avg_qty), 4),
            "size_vs_avg":    f"{row['quantity']/avg_qty:.1f}x",
            "prev_loss":      round(float(row["prev_pl"]), 2),
            "mins_after_loss": round(float(row["time_since_prev"]), 1),
        })

    count = len(result["flagged_trades"])
    result["details"]["revenge_trade_count"] = count
    result["details"]["time_window_minutes"] = time_window_min
    result["details"]["qty_multiplier_used"] = qty_multiplier

    if count > 0:
        result["flagged"] = True
        result["reasons"].append(
            f"Found {count} revenge trade(s): position size was ≥{qty_multiplier}x the "
            f"average AND opened within {time_window_min} min of a loss."
        )

    return result


# ─────────────────────────────────────────────────────────────
# CONVENIENCE: run all three at once
# ─────────────────────────────────────────────────────────────
def run_all(
    df: pd.DataFrame,
    max_per_hour:     int   = DEFAULT_MAX_TRADES_PER_HOUR,
    max_vol_ratio:    float = DEFAULT_VOL_BALANCE_RATIO,
    loss_win_ratio:   float = DEFAULT_LOSS_WIN_RATIO,
    revenge_mult:     float = DEFAULT_REVENGE_QTY_MULT,
    revenge_time_min: int   = DEFAULT_REVENGE_TIME_MIN,
) -> dict:
    """Run all three detectors and return a combined dict."""
    return {
        "overtrading":    detect_overtrading(df,  max_per_hour=max_per_hour,  max_vol_ratio=max_vol_ratio),
        "loss_aversion":  detect_loss_aversion(df, ratio_threshold=loss_win_ratio),
        "revenge_trading": detect_revenge_trading(df, qty_multiplier=revenge_mult, time_window_min=revenge_time_min),
    }