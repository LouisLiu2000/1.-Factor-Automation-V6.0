VOLTREND_ADV = rank(zscore(
  clip(
    (
      (
        (close / (ts_mean(close,24) + 0.000001)) - 1
        -
        (close / (ts_mean(close,96) + 0.000001) - 1)
      )
      *
      clip(
        (ts_std(logret(close,1),18) / (ts_std(logret(close,1),96) + 0.000001))
        *
        (ts_mean(abs(logret(close,1)),18) / (ts_mean(abs(logret(close,1)),96) + 0.000001)),
        0,
        3
      )
      *
      (ts_mean(volume,18) / (ts_mean(volume,96) + 1))
      *
      where(((close / (ts_mean(close,96) + 0.000001)) - 1) > 0, 1, 0.25)
    ),
    -6,
    6
  ),
  200
))
