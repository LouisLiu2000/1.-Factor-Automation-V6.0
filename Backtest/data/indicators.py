"""Collection of reusable indicator helpers for factor preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def money_flow_index(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute Money Flow Index (MFI).

    Parameters
    ----------
    data:
        Price DataFrame with at least ``high``, ``low``, ``close`` and ``volume`` columns.
    window:
        Rolling window length. Defaults to 14.
    """

    if window <= 0:
        raise ValueError("window must be positive")
    required = {"high", "low", "close", "volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"money_flow_index requires columns: {sorted(required)}; missing {sorted(missing)}")

    typical_price = (data["high"] + data["low"] + data["close"]) / 3.0
    money_flow = typical_price * data["volume"]
    delta = typical_price.diff()

    positive_flow = np.where(delta > 0, money_flow, 0.0)
    negative_flow = np.where(delta < 0, money_flow, 0.0)

    pos_sum = pd.Series(positive_flow, index=data.index).rolling(window=window, min_periods=1).sum()
    neg_sum = pd.Series(negative_flow, index=data.index).rolling(window=window, min_periods=1).sum()

    ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + ratio))
    mfi = mfi.fillna(50.0).rename("money_flow_index")
    return mfi
