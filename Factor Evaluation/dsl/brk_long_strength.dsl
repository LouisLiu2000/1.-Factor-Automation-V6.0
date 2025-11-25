BRK_LONG_STRENGTH = ts_rank(
    zscore((ts_mean(close,100) - ts_min(low,100)) / (ts_std(logret(close,1),60) + 0.000000001), 200)
  + 0.5*zscore((ts_mean(close,100) - shift(ts_mean(close,100),1)) / (ts_std(logret(close,1),60) + 0.000000001), 200)
, 200, 1)
