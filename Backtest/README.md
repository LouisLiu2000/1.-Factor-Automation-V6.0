# Backtest Engine

本项目提供一套基于 Backtrader 的截面因子回测体系，强调**数据可追溯、流程可复现、接口可扩展**。整体工程拆分为两个阶段：

- **阶段 A：数据准备与因子预计算（`run_prepare.py`）**
- **阶段 B：策略回测（`run_backtest.py`）**

两阶段共用同一套参数入口（timeframe、symbols、start/end、factor_set 等），确保预计算产物可以直接驱动回测。

## 目录结构

```
Backtest/
├── config/                # 引擎与策略的 YAML 配置
├── data/                  # 数据加载与 DSL 解析工具
├── engine/                # 回测执行、日志与参数扫描
├── results/               # 运行结果（索引 + 单次回测输出）
├── strategies/            # 策略实现：TopK / LongShort / Threshold
├── tests/                 # 基于 Pytest 的合成数据测试
├── run_prepare.py         # 阶段 A：数据准备 + 因子预计算 CLI
└── run_backtest.py        # 阶段 B：回测 CLI
```

## 阶段 A：数据准备与因子预计算

原始行情来自 `C:\Users\User\Desktop\Binance Data V3.0\6.Binance Data Resampled/<timeframe>/`，文件名包含起始毫秒时间戳，列格式固定：

```
open_time,open,high,low,close,volume,close_time,quote_asset_volume,
number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,
premium_open,premium_high,premium_low,premium_close,premium_close_time
```

使用 DSL（YAML/JSON）描述需要预计算的因子，示例：

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

> expression 由 `step3_factor_engine_04.py` 的 DSL 解析器校验与执行；`library_function` 可引用现有 Python 函数。

运行阶段 A：

```bash
python run_prepare.py \
  --start 2021-01-01 \
  --end 2025-01-01 \
  --timeframe 2H \
  --symbols BTCUSDT,ETHUSDT \
  --factor-set mfi14_v1 \
  --dsl-path factor_recipe.yaml \
  --data-root "C:/Users/User/Desktop/Binance Data V3.0" \
  --datahub-root DataHub \
  --log-level INFO
```

输出结果：

- `DataHub/factors/<timeframe>/<factor_set>/<symbol>.parquet`：包含 `timestamp`、所有因子列与 `tradable_flag`。
- `DataHub/metadata/<timeframe>/<factor_set>.json`：记录因子列、覆盖时间段、DSL 指纹、缺失值比例等。
- 日志写入 `FactorFoundry/logs/prepare_<factor_set>_*.log`。
- 终端打印 JSON 总览（每个币种的行数、缺失率等）。

## 阶段 B：策略回测

在阶段 A 成功落地因子后，直接调用：

```bash
python run_backtest.py \
  --start 2021-01-01 \
  --end 2025-01-01 \
  --timeframe 2H \
  --symbols BTCUSDT,ETHUSDT \
  --factor-set mfi14_v1 \
  --strategy topk \
  --param factor=mfi14 \
  --param k=5 \
  --param weighting=score \
  --dsl-path factor_recipe.yaml
```

CLI 会读取 `config/core.yaml` 与 `config/strategies/<strategy>.yaml` 的默认设置，可通过 `--param key=value` 临时覆盖参数。回测结果写入 `Backtest/results/<strategy>/<run_id>/`，包含：

- `metrics.json`：年化收益、波动、Sharpe、最大回撤、交易统计等。
- `equity_curve.csv`、`positions.csv`、`orders.csv`：权益曲线和持仓/交易明细。
- `run_context.json`：完整运行上下文（参数、数据位置、环境信息、因子 metadata）。
- `factor_metadata.json`：阶段 A 输出的 metadata 原封写入，便于审计。
- `logs.txt`：运行日志。

顶层 `Backtest/results/index.json` 会按 run 记录摘要，便于快速索引。

如果想快速查看某次运行的每日权益、收益分布与参数信息，可执行：

```bash
python -m Backtest.analyze_run --run-id <run_id> [--results-root Backtest/results]
```

例如：

```bash
python -m Backtest.analyze_run --run-id topk_60min_62_2c528388bb_mfi14_topk_test
```

脚本会读取对应目录，输出最近的每日权益、日收益率分布、最大回撤百分比以及策略/因子信息摘要，
并在结果目录生成 `*_daily_summary.csv`（包含每日收益与回撤百分比）。
若希望同时生成图形，可追加 `--plot`（可选 `--plot-dir` 指定输出目录）：

```bash
python -m Backtest.analyze_run --run-id <run_id> --plot
```

## 数据校验与安全性

- 时间戳统一转换为 UTC，并校验重复/缺失。
- DSL 表达式由 `step3_factor_engine_04.py` 的解析器验证，禁止未授权函数调用。
- 写入操作默认不覆盖旧文件，除非显式传入 `--overwrite`。
- metadata 记录因子列、生成时间、DSL 指纹（SHA1）、缺失率与样本数。

## 测试与依赖

建议在虚拟环境中安装依赖（Backtrader、pandas、PyYAML、pytest、pyarrow 等），然后执行：

```bash
pip install -r Backtest/requirements.txt
pytest Backtest/tests
```

测试使用合成 CSV/Parquet，覆盖 DataBundle、策略权重逻辑、Runner 流程，确保关键路径在无真实行情的情况下也可验证。
