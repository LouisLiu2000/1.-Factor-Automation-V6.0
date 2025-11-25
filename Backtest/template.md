
# Full Backtest Template

本模板涵盖因子预计算、回测执行、结果分析三大阶段。将占位符替换为实际值即可复用。

---

## 阶段 A：数据准备（因子预计算）

### 1. 需要确认的参数
| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `--start` / `--end` | 数据覆盖起止日期 | `2021-01-01` / `2025-01-01` |
| `--timeframe` | K 线周期，对应 `6.Binance Data Resampled/<timeframe>` | `4H`, `2H`, `60min` |
| `--symbols` | 逗号分隔的币种，无空格 | `BTCUSDT,ETHUSDT,...` |
| `--factor-set` | 因子版本名称 | `mfi14_v1` |
| `--dsl-path` | 因子 DSL 配方路径 | `Backtest/dsl/mfi14_v1.yaml` |
| `--data-root` | 原始行情目录 | `C:/Users/User/Desktop/Binance Data V3.0` |
| `--datahub-root` | 因子输出目录 | `DataHub` |

### 2. DSL 配方示例（Backtest/dsl/`<factor_set>`.yaml）
```yaml
default_na_threshold: 0.2
factors:
  - name: money_flow_index_14
    type: library_function
    source: Backtest.data.indicators:money_flow_index
    params:
      window: 14
    fillna: 50.0
  # 继续追加其他因子...
```

### 3. 执行命令
```powershell
python -m Backtest.run_prepare `
  --start YYYY-MM-DD `
  --end YYYY-MM-DD `
  --timeframe <TIMEFRAME> `
  --symbols SYMBOL1,SYMBOL2,... `
  --factor-set <FACTOR_SET> `
  --dsl-path Backtest/dsl/<FACTOR_SET>.yaml `
  --data-root "C:\Users\User\Desktop\Binance Data V3.0" `
  --datahub-root DataHub `
  --log-level INFO
```

### 4. 产物
- `DataHub/factors/<timeframe>/<factor_set>/<symbol>.parquet`
- `DataHub/metadata/<timeframe>/<factor_set>.json`
- 日志：`FactorFoundry/logs/prepare_<factor_set>_*.log`
- 终端显示因子质量统计（缺失值比例、行数等）

---

## 阶段 B：策略回测

### 1. 需要确认的参数
| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `--strategy` | 策略类型 | `topk`, `longshort`, `threshold`, 其他自定义 |
| **TopK 专属** |||
| `--param factor=<因子列>` | 使用的因子列 | `momentum_6h` |
| `--param k=<整数>` | 保留的标的数量 | `5` |
| `--param weighting=<equal/score>` | 权重方式 | `equal` |
| `--param min_factor=<数值>` | (可选) 因子阈值 | `0.5` |
| **LongShort 专属** |||
| `--param factor=<因子列>` | 使用的因子列 | `momentum_6h` |
| `--param long_quantile=<0~1>` | 做多分位 | `0.2` |
| `--param short_quantile=<0~1>` | 做空分位 | `0.2` |
| `--param gross_exposure=<数值>` | 多空合计杠杆 | `1.0` |
| `--param weighting=<equal/score>` | 权重方式 | `score` |
| `--param hedge_ratio=<数值>` | 空头与多头比率 | `1.0` |
| **Threshold 专属** |||
| `--param factor=<因子列>` | 使用的因子列 | `momentum_6h` |
| `--param buy_threshold=<数值>` | 开多阈值 | `0.02` |
| `--param sell_threshold=<数值>` | 平多/开空阈值 | `-0.02` |
| `--param position_size=<0~1>` | 持仓占比 | `0.1` |
| `--param hold_max_bars=<整数>` | (可选) 持仓最久 bar 数 | `24` |
| `--param allow_short=<true/false>` | 是否允许做空 | `true` |
| **公共参数** | `start`, `end`, `timeframe`, `symbols`, `factor-set`, `dsl-path`, `results-tag` 等 | 与阶段 A 相同 |

> 若策略有额外参数，全部通过 `--param key=value` 形式注入。例如：
> ```powershell
> --param max_positions=20 --param cash_buffer=0.05
> ```

### 2. 执行命令
```powershell
python -m Backtest.run_backtest `
  --start YYYY-MM-DD `
  --end YYYY-MM-DD `
  --timeframe <TIMEFRAME> `
  --symbols SYMBOL1,SYMBOL2,... `
  --factor-set <FACTOR_SET> `
  --strategy topk `
  --param factor=money_flow_index_14 `
  --param k=10 `
  --param weighting=score `
  --dsl-path Backtest/dsl/<FACTOR_SET>.yaml `
  --data-root "C:\Users\User\Desktop\Binance Data V3.0" `
  --datahub-root DataHub `
  --output-root Backtest/results `
  --results-tag <RUN_TAG> `
  --log-level INFO
```

### 3. 产物（位于 `Backtest/results/<strategy>/<run_id>/`）
- `metrics.json`：回测关键指标
- `equity_curve.csv`：每根 bar 的收益/权益
- `positions.csv` / `orders.csv`：持仓与订单明细
- `run_context.json`：完整运行参数、环境
- `factor_metadata.json`：所用因子版本的 metadata 备份
- `logs.txt`：运行日志
- 顶层 `Backtest/results/index.json` 会新增一条 run 记录

---

## 阶段 C：分析与可视化

### 1. 快速查看
```powershell
python -m Backtest.analyze_run --run-id <RUN_ID>
```
输出内容：
- 策略、因子集、时间框、符号列表
- `metrics.json` 中的关键指标
- 最近 10 天的日权益、日收益（%）、日回撤（%）
- 日收益率分布（区间统计）
- `*_daily_summary.csv`：包含每日收益率、每日回撤（%）、累计收益等详表

### 2. 生成图表
```powershell
python -m Backtest.analyze_run --run-id <RUN_ID> --plot [--plot-dir <路径>]
```
- 输出一张 `..._analysis.png`，含每日权益、日收益、日回撤曲线
- 默认保存到 run 目录，若指定 `--plot-dir` 则保存到目标文件夹
- 需要系统已安装 `matplotlib`

---

## 快速填表

| 配置项 | 取值 |
| --- | --- |
| timeframe | `<TIMEFRAME>` |
| start / end | `<START>` / `<END>` |
| symbols | `<SYMBOL_LIST>` |
| factor_set | `<FACTOR_SET>` |
| DSL 配方 | `Backtest/dsl/<FACTOR_SET>.yaml` |
| strategy | `topk / longshort / threshold / ...` |
| strategy params | `factor=...`, `k=...`, `weighting=...`, ... |
| run tag | `<RUN_TAG>` |

按照以上模板填入参数，即可完成“因子准备 → 策略回测 → 结果分析”的全流程。祝顺利！♪
