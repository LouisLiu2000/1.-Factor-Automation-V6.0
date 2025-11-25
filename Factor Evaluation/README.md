# Factor Automation V5.0

Factor Automation V5.0 用于在多种 K 线频率下准备、生成并评估 Binance 重采样数据与 Qlib 因子。当前所有脚本均支持 `--bar-freq` 参数，可按需选择 `1min`、`3min`、`5min`、`10min`、`15min`、`30min`、`1H`、`2H`、`4H` 等 pandas 频率字符串。

## 频率配置
- 未显式指定 `--bar-freq` 时默认使用 `1min`，输出会写入 `<run_id>/1min/…`，日志位于 `logs/1min/`。
- 原始重采样数据放在 `C:\Users\User\Desktop\Binance Data V3.0\6.Binance Data Resampled\<bar-freq>\` 中；若缺失会尝试回退到 `4.Binance Data Merged\Data` 并给出 WARN，仍缺失则终止执行。
- 所有 horizon 参数依旧表示“若干根 bar”，实际时间跨度通过 `offset * horizon` 计算。
- 因子评估阶段的标签统一为对数收益率（log return），便于跨周期累加和统计分析。
- 输出、缓存、日志、报告等均按频率隔离，便于并行生成多套结果。

## 常用命令
```
python step0_prepare_inputs.py --data-root "C:\Users\User\Desktop\Binance Data V3.0" --output-root "C:\Users\User\Desktop\AI_Agent_Output_fix" --run-id demo_run --start 2025-01-01 --end 2025-01-03 --symbols BTCUSDT --mode execute
python step1_contract_and_golden.py --data-root "C:\Users\User\Desktop\Binance Data V3.0" --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --symbol BTCUSDT --date 2025-01-02
```

## 3 分钟全流程示例
```
python step0_prepare_inputs.py           --data-root "C:\Users\User\Desktop\Binance Data V3.0" --output-root "C:\Users\User\Desktop\AI_Agent_Output_fix" --run-id momentum_2025 --start 2025-01-23 --end 2025-09-01 --symbols BTC,ETH --bar-freq 3min --mode execute
python step0_prepare_inputs_qlib_01.py   --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --run-id momentum_2025 --bar-freq 3min
python step1_contract_and_golden.py      --data-root "C:\Users\User\Desktop\Binance Data V3.0" --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --symbol BTCUSDT --date 2025-01-24 --bar-freq 3min
python step2_loader_and_qc.py            --data-root "C:\Users\User\Desktop\Binance Data V3.0" --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --run-id momentum_2025 --start-date 2025-01-23 --end-date 2025-09-01 --bar-freq 3min
python step3_factor_engine_04.py         --data-root "C:\Users\User\Desktop\Binance Data V3.0" --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix\momentum_2025" --run-id momentum_2025 --symbols BTCUSDT --bar-freq 3min --start 2025-01-23 --end 2025-09-01 --dsl "MOM3_SIMPLE = close / shift(close, 3) - 1"
python step4_factor_evaluation.py        --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --factor-name MOM3_SIMPLE --run-id momentum_2025 --bar-freq 3min --horizons 5,15
```
执行完后可在 `AI_Agent_Output_fix\momentum_2025\logs\3min` 查看 `bar_freq=3min` 相关日志。

## 测试
```
python -m pytest test_qlib_integration.py::test_qlib_pipeline
```
该测试会分别运行 1min 与 3min 的缩减版流水线，并断言日志内不存在频率串联错误。

## 常见问题
- **数据目录缺失**：确认 `6.Binance Data Resampled\<bar-freq>` 已存在；若回退至 `4.Binance Data Merged\Data` 也缺失则需补齐数据。
- **日志检查**：多频率执行时务必检查 `logs/<bar_freq>/`，确保没有 `bar_freq=1min` 等残留信息。
- **回退提示**：若看到 WARN 提示使用 legacy 路径，说明当前频率的重采样数据暂缺，可按需补齐或接受旧数据。
- **默认行为**：未传 `--bar-freq` 时保持 1 分钟流程，不影响历史脚本使用方式。

更多细节请参考 `Qlib_pipeline.md`。
## 5 分钟多币种示例
以下示例基于 `run-id=backtest_5min`，一次性覆盖 50 个符号（BTC、ETH、…、1000FLOKI）。步骤全部完成后，可在 `AI_Agent_Output_fix\backtest_5min` 下看到完整的 `golden/5min`、`qlib/5min`、`reports_qlib/5min` 目录。

1. **Step0 生成任务（直接执行）**
   ```powershell
   python step0_prepare_inputs.py `
     --project-root "C:\Users\User\Desktop\Factor Automation V5.0" `
     --data-root    "C:\Users\User\Desktop\Binance Data V3.0" `
     --output-root  "C:\Users\User\Desktop\AI_Agent_Output_fix" `
     --run-id backtest_5min `
     --start 1690597800000 `
     --end   1756512000000 `
     --symbols BTC,ETH,SOL,XRP,DOGE,1000PEPE,SUI,ADA,TRUMP,BNB,WIF,ENA,NEIRO,LINK,FARTCOIN,PNUT,LTC,AVAX,WLD,AAVE,1000SHIB,1000BONK,XLM,HBAR,MOODENG,UNI,TIA,DOT,PENGU,BCH,ONDO,APT,NEAR,CRV,FIL,TRX,TAO,ORDI,ARB,POPCAT,OP,FET,SEI,VIRTUAL,ACT,OM,GOAT,KAITO,ETC,1000FLOKI `
     --bar-freq 5min `
     --horizons 5,15,30,60 `
     --mode execute `
     --concurrency 6
   ```
   *TIP：对于上市较晚的币种，`first_open_time_ms` 会晚于 `--start`，这些符号在 Step0 manifest 中会标记为 `first_trade_after_window` 并不会生成黄金样本，后续 Step4 Qlib 会提示 “Label file missing”。如需纳入评估，可在实际上市日之后重新运行 Step0~Step1。*

2. **Step0 → Qlib 转换**
   ```powershell
   python step0_prepare_inputs_qlib_01.py --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --run-id backtest_5min --bar-freq 5min
   ```

3. **Step1 Qlib 转换**（Step0 已执行任务，可直接将黄金样本写入 Qlib）
   ```powershell
   python step1_contract_and_golden_qlib_02.py --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --run-id backtest_5min --bar-freq 5min
   ```

4. **Step2 覆盖率 / Step2 Qlib**
   ```powershell
   python step2_loader_and_qc.py          --data-root "C:\Users\User\Desktop\Binance Data V3.0" --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --run-id backtest_5min --start-date 2023-07-28 --end-date 2025-08-30 --symbols BTC,...,1000FLOKI --bar-freq 5min
   python step2_loader_and_qc_qlib_03.py  --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --run-id backtest_5min --bar-freq 5min
   ```

5. **Step3 计算波动率因子（VOL_BURST_COMPLEX）**
   ```powershell
   python step3_factor_engine_04.py `
     --data-root "C:\Users\User\Desktop\Binance Data V3.0" `
     --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix\backtest_5min" `
     --run-id backtest_5min `
     --symbols BTC,ETH,SOL,...,1000FLOKI `
     --bar-freq 5min `
     --start 2023-07-28 `
     --end   2025-08-30 `
     --dsl "VOL_BURST_COMPLEX = (ts_std(logret(close, 1), 12) / (ts_std(logret(close, 1), 96) + 0.000001) * ts_mean(abs(logret(close, 1)), 12) / (ts_mean(abs(logret(close, 1)), 96) + 0.000001) * ts_std(logret(close, 1), 12) / (ema(ts_std(logret(close, 1), 12), 24) + 0.000001))"
   ```

6. **Step4 Qlib 评估 + 分析**
   ```powershell
   python step4_factor_evaluation_qlib_05.py `
     --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" `
     --factor-name VOL_BURST_COMPLEX `
     --run-id backtest_5min `
     --bar-freq 5min `
     --horizons 5,15,30,60 `
     --group 5 `
     --time-split month `
     --comparison-threshold 0.05 `
     --skip-local `
     --enable-report-analysis `
     --analysis-output-dir analysis `
     --analysis-format html
   ```

重跑 Step3/Step4 后会覆盖同名因子（`factors/5min/VOL_BURST_COMPLEX`、`qlib/5min/features/VOL_BURST_COMPLEX`），同时更新 `reports_qlib` 与 `config_snapshots`。若只需要少量符号，可以将 `--symbols` 换成对应子集，以减少计算量。

