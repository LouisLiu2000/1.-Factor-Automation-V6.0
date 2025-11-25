# Factor Automation V6.0

一套从 Binance 重采样数据出发，完成**因子生成 → 评估 → 截面回测**的端到端工具链。项目拆分为两条流水线：

- **Factor Evaluation**：多频率因子计算与 Qlib/统计评估（阶段 0~4）。
- **Backtest**：基于 Backtrader 的截面因子回测引擎（阶段 A/B）。

整体设计强调数据可追溯、流程可复现、接口可扩展：输入/输出路径和参数均可通过 CLI 控制，因子定义支持 DSL 与自定义 Python 函数。

## 仓库结构
- `Backtest/`：截面回测引擎、CLI（`run_prepare.py`/`run_backtest.py`）、配置、策略、测试。
- `Factor Evaluation/`：因子生成与评估脚本（Step0~Step4）、Qlib 集成、报告工具。
- `DataHub/`：阶段 A 产物（`factors/<timeframe>/<factor_set>/*.parquet`、`metadata/*.json`）。
- `FactorFoundry/logs/`：阶段 A/B 的运行日志目录。
- `README.md`：当前文件。

## 环境与依赖
建议使用 Python 3.10+ 和虚拟环境。分别安装两条流水线的依赖：

```bash
# 截面回测
python -m venv .venv && source .venv/bin/activate
pip install -r Backtest/requirements.txt

# 因子生成与评估
pip install -r "Factor Evaluation/requirements.txt"
```

## 数据准备
- 默认原始重采样数据位置：`C:/Users/User/Desktop/Binance Data V3.0/6.Binance Data Resampled/<timeframe>/`。
- CSV 列格式固定：`open_time,open,high,low,close,volume,close_time,quote_asset_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,premium_open,premium_high,premium_low,premium_close,premium_close_time`。
- 回测阶段使用的因子由 DSL 描述，可放在 YAML/JSON（示例见下文）。所有时间戳统一转换为 UTC 并校验重复/缺失。

## 快速开始：截面因子回测（Backtest）
阶段 A/B 共用一套参数入口，保证预计算的因子可直接驱动回测。

1) 准备因子 DSL（`factor_recipe.yaml`）
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

2) 阶段 A：数据准备与因子预计算
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
输出：
- `DataHub/factors/<timeframe>/<factor_set>/<symbol>.parquet`：`timestamp`、因子列、`tradable_flag`。
- `DataHub/metadata/<timeframe>/<factor_set>.json`：因子列、覆盖时间段、DSL SHA1、缺失率等。
- 日志写入 `FactorFoundry/logs/prepare_<factor_set>_*.log`，终端打印 JSON 总览。

3) 阶段 B：策略回测
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
结果写入 `Backtest/results/<strategy>/<run_id>/`，包含 `metrics.json`、`equity_curve.csv`、`positions.csv`、`orders.csv`、`run_context.json`、`factor_metadata.json`、`logs.txt`。索引汇总位于 `Backtest/results/index.json`。

4) 快速查看某次运行
```bash
python -m Backtest.analyze_run --run-id <run_id> [--results-root Backtest/results] [--plot]
```

## 因子生成与评估流水线（Factor Evaluation）
面向多频率因子计算、覆盖率检查、报告输出（Qlib 兼容）。核心步骤：
- **Step0** `step0_prepare_inputs.py`：生成任务清单、黄金样本、标签；支持并发与过滤，默认输出 `AI_Agent_Output_fix/<run_id>/<bar_freq>/`。
  - **Step0 → Qlib** `step0_prepare_inputs_qlib_01.py`：将 Step0 产物转换为 Qlib 结构（`qlib/<bar_freq>/features` 等）。
- **Step1** `step1_contract_and_golden*.py`：将黄金样本写入标准/qlib 特征（`*_qlib_02.py` 为 Qlib 版本）。
- **Step2** `step2_loader_and_qc*.py`：覆盖率和质量检查，可生成报告；`*_qlib_03.py` 为 Qlib 版本。
- **Step3** `step3_factor_engine_04.py`：执行 DSL 因子计算（语法与 Backtest 共用），生成因子列。
- **Step4** `step4_factor_evaluation*.py`：分组/分层评估，输出统计或 Qlib 报告；`*_qlib_05.py` 支持 Qlib 报告/图表。

示例（3 分钟频率，全流程缩略版）：
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
日志默认写入 `<output-root>/<run_id>/logs/<bar_freq>/`。

Qlib 转换 & 报告示例（5 分钟，多币种）：
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

Qlib 目录结构（位于 `<output-root>/<run_id>`）：
- `golden/<bar_freq>/`：黄金样本与标签。
- `qlib/<bar_freq>/features/`：Qlib 特征与标签。
- `factors/<bar_freq>/`：Step3 输出的因子列。
- `reports_qlib/<bar_freq>/`：Step4 评估报告与图表。
- `config_snapshots/<bar_freq>/`：运行时配置快照。
- `logs/<bar_freq>/`：流水线日志。

## 输出与日志一览
- `DataHub/`：阶段 A 因子与 metadata，供回测使用。
- `Backtest/results/`：策略运行的度量、曲线、订单、上下文与索引。
- `FactorFoundry/logs/`：阶段 A/B 日志。
- `AI_Agent_Output_fix/<run_id>/...`（可自定义）：Factor Evaluation 的黄金样本、因子、Qlib 特征、报告与日志。

## 测试
```bash
pytest Backtest/tests
pytest "Factor Evaluation/test_qlib_integration.py"::test_qlib_pipeline
```

## 进阶与备注
- CLI 参数可通过 `-h` 查看完整列表；`config/core.yaml` 与 `config/strategies/<strategy>.yaml` 提供回测默认值，可用 `--param key=value` 临时覆盖。
- DSL 支持 `expression` 与 `library_function`，解析器位于 `Backtest/data/dsl`，自定义函数需在 Python 模块内注册后引用。
- 写入操作默认不覆盖旧文件，如需重算传入 `--overwrite`（阶段 A/B）或对应脚本参数。
- 若遇到数据缺失/频率不匹配，优先检查原始数据目录与日志提示。
