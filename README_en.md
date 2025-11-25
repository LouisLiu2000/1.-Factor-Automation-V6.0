# Factor Automation V6.0

End-to-end toolchain that builds **factors → evaluates them → cross-sectional backtests** on Binance resampled data. Two pipelines:

- **Factor Evaluation**: multi-frequency factor generation, coverage checks, and Qlib/statistical evaluation (Steps 0–4).
- **Backtest**: Backtrader-based cross-sectional engine (Stages A/B).

Design goals: reproducible I/O, auditability, and extensibility. All inputs/outputs are CLI-driven; factors can be declared via DSL or Python functions.

## Repository Layout
- `Backtest/`: backtest engine, CLIs (`run_prepare.py` / `run_backtest.py`), configs, strategies, tests.
- `Factor Evaluation/`: factor generation & evaluation scripts (Step0–Step4), Qlib integration, reporting.
- `DataHub/`: Stage A outputs (`factors/<timeframe>/<factor_set>/*.parquet`, `metadata/*.json`).
- `FactorFoundry/logs/`: logs for Stage A/B.
- `README.md`: Chinese version; `README_en.md`: this file.

## Environment & Dependencies
Use Python 3.10+ with a virtual environment. Install per pipeline:
```bash
# Cross-sectional backtest
python -m venv .venv && source .venv/bin/activate
pip install -r Backtest/requirements.txt

# Factor Evaluation
pip install -r "Factor Evaluation/requirements.txt"
```

## Data Expectations
- Default resampled data: `C:/Users/User/Desktop/Binance Data V3.0/6.Binance Data Resampled/<timeframe>/`.
- CSV columns: `open_time,open,high,low,close,volume,close_time,quote_asset_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,premium_open,premium_high,premium_low,premium_close,premium_close_time`.
- Factor definitions live in YAML/JSON DSL (example below). Timestamps are normalized to UTC; duplicates/missing values are validated.

## Quickstart: Cross-Sectional Backtest (Backtest)
Stage A/B share the same parameter interface so precomputed factors feed backtests directly.

1) Factor DSL (`factor_recipe.yaml`)
```yaml
factors:
  - name: mfi14
    type: expression
    expression: "where(volume > 0, ts_sum((high+low+close)/3 * volume, 14) / ts_sum(volume, 14), 0)"
    na_threshold: 0.2
  - name: money_flow_index
    type: library_function
    source: Backtest.data.indicators:money_flow_index
    params:
      window: 14
```

2) Stage A: data prep & factor pre-computation
```bash
python Backtest/run_prepare.py \
  --start 2021-01-01 --end 2025-01-01 \
  --timeframe 2H \
  --symbols BTCUSDT,ETHUSDT \
  --factor-set mfi14_v1 \
  --dsl-path factor_recipe.yaml \
  --data-root "C:/Users/User/Desktop/Binance Data V3.0" \
  --datahub-root DataHub \
  --log-level INFO
```
Outputs:
- `DataHub/factors/<timeframe>/<factor_set>/<symbol>.parquet`: `timestamp`, factor columns, `tradable_flag`.
- `DataHub/metadata/<timeframe>/<factor_set>.json`: factor columns, coverage window, DSL SHA1, missing ratios, row counts.
- Logs in `FactorFoundry/logs/prepare_<factor_set>_*.log`; a JSON summary is printed to terminal.

3) Stage B: backtest
```bash
python Backtest/run_backtest.py \
  --start 2021-01-01 --end 2025-01-01 \
  --timeframe 2H \
  --symbols BTCUSDT,ETHUSDT \
  --factor-set mfi14_v1 \
  --strategy topk \
  --param factor=mfi14 \
  --param k=5 \
  --param weighting=score \
  --dsl-path factor_recipe.yaml
```
Results written to `Backtest/results/<strategy>/<run_id>/`: `metrics.json`, `equity_curve.csv`, `positions.csv`, `orders.csv`, `run_context.json`, `factor_metadata.json`, `logs.txt`. Run index lives in `Backtest/results/index.json`.

4) Inspect a run quickly
```bash
python -m Backtest.analyze_run --run-id <run_id> [--results-root Backtest/results] [--plot]
```

## Factor Generation & Evaluation (Factor Evaluation)
Targets multi-frequency factor computation, coverage QC, and reporting (Qlib-ready). Core steps:
- **Step0** `step0_prepare_inputs.py`: build task manifest, golden samples, labels; supports concurrency/filters. Default output `AI_Agent_Output_fix/<run_id>/<bar_freq>/`.
  - **Step0 → Qlib** `step0_prepare_inputs_qlib_01.py`: convert Step0 outputs to Qlib layout (`qlib/<bar_freq>/features`, etc.).
- **Step1** `step1_contract_and_golden*.py`: write golden samples / Qlib features (`*_qlib_02.py` for Qlib).
- **Step2** `step2_loader_and_qc*.py`: coverage & quality checks, optional reports (`*_qlib_03.py` for Qlib).
- **Step3** `step3_factor_engine_04.py`: run DSL factor calculations (same syntax as Backtest), produce factor columns.
- **Step4** `step4_factor_evaluation*.py`: grouped/stratified evaluation; `*_qlib_05.py` emits Qlib reports/plots.

Minimal 3min example:
```bash
python "Factor Evaluation/step0_prepare_inputs.py" --data-root "C:/Users/User/Desktop/Binance Data V3.0" \
  --output-root "C:/Users/User/Desktop/AI_Agent_Output_fix" --run-id demo_run \
  --start 2025-01-01 --end 2025-01-03 --symbols BTCUSDT --bar-freq 3min --mode execute

python "Factor Evaluation/step3_factor_engine_04.py" \
  --data-root "C:/Users/User/Desktop/Binance Data V3.0" \
  --output-dir "C:/Users/User/Desktop/AI_Agent_Output_fix/demo_run" \
  --run-id demo_run --symbols BTCUSDT --bar-freq 3min --start 2025-01-01 --end 2025-01-03 \
  --dsl "MOM3_SIMPLE = close / shift(close, 3) - 1"

python "Factor Evaluation/step4_factor_evaluation.py" \
  --output-dir "C:/Users/User/Desktop/AI_Agent_Output_fix" \
  --factor-name MOM3_SIMPLE --run-id demo_run --bar-freq 3min --horizons 5,15
```
Logs default to `<output-root>/<run_id>/logs/<bar_freq>/`.

Qlib conversion & reporting (5min, multi-symbol):
```bash
python "Factor Evaluation/step0_prepare_inputs.py" \
  --data-root "C:/Users/User/Desktop/Binance Data V3.0" \
  --output-root "C:/Users/User/Desktop/AI_Agent_Output_fix" \
  --run-id backtest_5min --start 2023-07-28 --end 2025-08-30 \
  --symbols BTC,ETH,SOL,XRP,DOGE --bar-freq 5min --horizons 5,15,30,60 --mode execute --concurrency 4

python "Factor Evaluation/step0_prepare_inputs_qlib_01.py" \
  --output-dir "C:/Users/User/Desktop/AI_Agent_Output_fix" --run-id backtest_5min --bar-freq 5min

python "Factor Evaluation/step1_contract_and_golden_qlib_02.py" \
  --output-dir "C:/Users/User/Desktop/AI_Agent_Output_fix" --run-id backtest_5min --bar-freq 5min

python "Factor Evaluation/step2_loader_and_qc_qlib_03.py" \
  --output-dir "C:/Users/User/Desktop/AI_Agent_Output_fix" --run-id backtest_5min --bar-freq 5min

python "Factor Evaluation/step3_factor_engine_04.py" \
  --data-root "C:/Users/User/Desktop/Binance Data V3.0" \
  --output-dir "C:/Users/User/Desktop/AI_Agent_Output_fix/backtest_5min" \
  --run-id backtest_5min --symbols BTC,ETH,SOL,XRP,DOGE \
  --bar-freq 5min --start 2023-07-28 --end 2025-08-30 \
  --dsl "VOL_BURST_COMPLEX = (ts_std(logret(close, 1), 12) / (ts_std(logret(close, 1), 96) + 0.000001) * ts_mean(abs(logret(close, 1)), 12) / (ts_mean(abs(logret(close, 1)), 96) + 0.000001) * ts_std(logret(close, 1), 12) / (ema(ts_std(logret(close, 1), 12), 24) + 0.000001))"

python "Factor Evaluation/step4_factor_evaluation_qlib_05.py" \
  --output-dir "C:/Users/User/Desktop/AI_Agent_Output_fix" \
  --factor-name VOL_BURST_COMPLEX --run-id backtest_5min --bar-freq 5min \
  --horizons 5,15,30,60 --group 5 --time-split month --analysis-format html --enable-report-analysis
```

Qlib directory layout (under `<output-root>/<run_id>`):
- `golden/<bar_freq>/`: golden samples and labels.
- `qlib/<bar_freq>/features/`: Qlib features/labels.
- `factors/<bar_freq>/`: Step3 factor outputs.
- `reports_qlib/<bar_freq>/`: Step4 evaluation reports/plots.
- `config_snapshots/<bar_freq>/`: runtime config snapshots.
- `logs/<bar_freq>/`: pipeline logs.

## Outputs & Logs
- `DataHub/`: Stage A factors and metadata for backtests.
- `Backtest/results/`: per-run metrics, curves, orders, contexts, and index.
- `FactorFoundry/logs/`: Stage A/B logs.
- `AI_Agent_Output_fix/<run_id>/...` (customizable): Factor Evaluation golden data, factors, Qlib features, reports, logs.

## Tests
```bash
pytest Backtest/tests
pytest "Factor Evaluation/test_qlib_integration.py"::test_qlib_pipeline
```

## Tips & Notes
- Use `-h` on each CLI for full parameters; `config/core.yaml` and `config/strategies/<strategy>.yaml` hold backtest defaults. Override with `--param key=value`.
- DSL supports `expression` and `library_function`; parser lives in `Backtest/data/dsl`. Register custom Python functions and reference them via `module:function`.
- Writes are non-overwriting by default; pass `--overwrite` (Stage A/B) or script-specific flags to recompute.
- If you see missing data or frequency mismatches, first inspect the source data directories and the logs.***
