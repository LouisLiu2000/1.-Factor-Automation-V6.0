A_TRV = rank(zscore(clip(
    -1 * (
        sign(ema(close, 24) - ema(close, 96))
        * (
            zscore(ts_sum(sign(logret(close, 1)), 18), 200)
            + (ema(close, 24) - ema(close, 96)) / (abs(ema(close, 96)) + 0.000001)
        )
        * (ts_std(logret(close, 1), 18) / (ts_std(logret(close, 1), 96) + 0.000001))
        * (
            ((rolling_max(high, 36) + rolling_min(low, 36)) / 2 - close)
            / (ema(abs(ret(close, 1)), 36) + 0.000001)
        )
    ),
    -6,
    6
), 200))
