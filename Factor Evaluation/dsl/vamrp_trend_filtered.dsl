VAMRP_TREND_FILTERED = rank(zscore(
  clip(
    where(
      (((rolling_max(high,36)+rolling_min(low,36))/2 - (rolling_max(high,72)+rolling_min(low,72))/2) /
        ((rolling_max(high,72)+rolling_min(low,72))/2) > 0),
      clip(
        (((rolling_max(high,36)+rolling_min(low,36))/2 - close) / ema(abs(ret(close,1)),36)) *
        (((rolling_max(high,36)+rolling_min(low,36))/2 + 1.5*ema(abs(ret(close,1)),36) - (rolling_max(high,72)+rolling_min(low,72))/2) /
          ((rolling_max(high,36)+rolling_min(low,36))/2 + 1.5*ema(abs(ret(close,1)),36))),
        0, 1e9),
      clip(
        (((rolling_max(high,36)+rolling_min(low,36))/2 - close) / ema(abs(ret(close,1)),36)) *
        (((rolling_max(high,36)+rolling_min(low,36))/2 + 1.5*ema(abs(ret(close,1)),36) - (rolling_max(high,72)+rolling_min(low,72))/2) /
          ((rolling_max(high,36)+rolling_min(low,36))/2 + 1.5*ema(abs(ret(close,1)),36))),
        -1e9, 0)),
    -5, 5), 200))
