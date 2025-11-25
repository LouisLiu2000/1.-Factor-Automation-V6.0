## Qlib 全流程操作指南

以下流程假设：
- 仓库路径：`C:\Users\User\Desktop\Factor Automation V3.0`
- Binance Fullversion 数据路径：`C:\Users\User\Desktop\Binance Data V3.0`
- 输出根目录：`C:\Users\User\Desktop\AI_Agent_Output_fix`
- 演示运行 ID：`momentum_2025`
- 运行环境：Windows PowerShell + Python 3.11

---

### 0. 环境准备
1. 打开 PowerShell 并进入仓库目录：
   ```powershell
   cd "C:\Users\User\Desktop\Factor Automation V3.0"
   ```
2. 安装依赖（首次或 requirements.txt 更新后执行）：
   ```powershell
   pip install -r requirements.txt
   ```

### 频率配置说明
- `--bar-freq` 支持 pandas 频率字符串（示例：`1min`、`3min`、`5min`、`10min`、`15min`、`30min`、`1H`、`2H`、`4H` 等）。
- 默认不传时沿用 `1min`，输出会记录在 `<run_id>/<bar_freq>/…` 子目录，并同步写入 `logs/<bar_freq>/` 下的日志文件。
- 原始数据位于 `C\Users\User\Desktop\Binance Data V3.0\6.Binance Data Resampled\<bar-freq>\`，当缺失时脚本会尝试回退到 `4.Binance Data Merged\Data` 并给出 WARN。
- 示例输出：`AI_Agent_Output_fix\momentum_2025\1min\step0_manifest.json`、`AI_Agent_Output_fix\momentum_2025\golden\3min\base_BTCUSDT_20240101.parquet`。
- 所有 horizon 参数仍表示“若干根 bar”，实际时间跨度由 `bar_freq` 的 offset * horizon 决定。
- 因子评估阶段的标签使用对数收益率（log return），默认写入 `label_ret_<horizon>`。
- 日志会标注使用的频率，便于排查多频率运行产生的覆盖或混用问题。

---

### 1. Step0：生成运行清单与任务计划
**脚本：`step0_prepare_inputs.py`**

| 参数 | 说明 |
| --- | --- |
| `--project-root` | 仓库根目录（例：`C:\Users\User\Desktop\Factor Automation V3.0`） |
| `--data-root` | Binance Fullversion 数据根目录（例：`C:\Users\User\Desktop\Binance Data V3.0`） |
| `--output-root` | 输出根目录（例：`C:\Users\User\Desktop\AI_Agent_Output_fix`） |
| `--run-id` | 运行 ID |
| `--start` / `--end` | 时间范围（毫秒时间戳或 ISO8601 均可） |
| `--symbols` | 逗号分隔的符号列表，`ALL` 表示全部 |
| `--bar-freq` | K线频率（符合 pandas offset 格式，默认 `1min`） |
| `--mode` | `plan` 或 `execute`; 建议直接使用 `execute` |
| `--concurrency` | 并发数（视机器性能调整） |
> 提示：脚本使用 `--output-root`（非 `--output-dir`）。如果需要重跑，加入 `--force` 覆盖旧数据。

示例：
```powershell
python step0_prepare_inputs.py `
  --project-root "C:\Users\User\Desktop\Factor Automation V3.0" `
  --data-root    "C:\Users\User\Desktop\Binance Data V3.0" `
  --output-root  "C:\Users\User\Desktop\AI_Agent_Output_fix" `
  --run-id momentum_2025 `
  --start "2025-01-23T17:00:00+00:00" `
  --end   "2025-09-01T00:00:00+00:00" `
  --symbols BTC,ETH,SOL,XRP,DOGE,1000PEPE,SUI,ADA,TRUMP,BNB,WIF,ENA,NEIRO,LINK,FARTCOIN,PNUT,LTC,AVAX,WLD,AAVE `
  --bar-freq 3min `
  --mode execute `
  --concurrency 7 `
  --force
```
输出：`step0_manifest.json`、`step0_summary.json`、`logs/step0_*` 等。

---

### 2. Step0 -> Qlib 转换
**脚本：`step0_prepare_inputs_qlib_01.py`**

| 参数 | 说明 |
| --- | --- |
| `--output-dir` | Step0 输出根目录（例：`C:\Users\User\Desktop\AI_Agent_Output_fix`） |
| `--run-id` | 运行 ID（例：`momentum_2025`） |
| `--manifest` / `--summary` | （可选）自定义路径 |
| `--bar-freq` | Qlib 输出使用的 K 线频率（默认 `1min`） |
| `--calendar-freq` | （可选）覆盖日历频率，默认与 `--bar-freq` 一致 |
| `--force` | 覆盖既有 Qlib 目录 |
> 注意：此脚本仅消费 Step0 已生成的 manifest/summary，不接受 `--project-root`、`--start`、`--symbols` 等参数，若要变更条件请重新执行 Step0。

```powershell
python step0_prepare_inputs_qlib_01.py `
  --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" `
  --run-id momentum_2025 `
  --bar-freq 3min `
  --force
```
输出：`qlib/calendar_{run_id}.csv`、`qlib/instruments_{run_id}.txt`（初版）、`qlib/segments_{run_id}.json` 等。

---

### 3. Step1：黄金数据转换为 Qlib 结构
**脚本：`step1_contract_and_golden_qlib_02.py`**

```powershell
python step1_contract_and_golden_qlib_02.py `
  --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" `
  --run-id momentum_2025 `
  --bar-freq 3min `
  --force
```
> 若已有旧输出，必须加 `--force`（或先清理对应文件）。

输出：`qlib/features/base/*.parquet`（多符号 multi-index）、`qlib/features/{factor}/` 初始标签、`config_snapshots/.../step1_qlib_summary.json` 等。

---

### 4. Step2：Universe 与覆盖评估（原始脚本）
**脚本：`step2_loader_and_qc.py`**

| 参数 | 说明 |
| --- | --- |
| `--data-root` | Binance Fullversion 数据根目录 |
| `--output-dir` | 输出根目录 |
| `--run-id` | 运行 ID |
| `--start-date` / `--end-date` | UTC 时间窗口（建议与 Step0 一致） |
| `--symbols` | 与 Step0 相同的符号列表或 `ALL` |
| `--bar-freq` | ??????? `1min`? |
| `--min-coverage` | 最低覆盖率（默认 0.8） |
| `--strict-readonly` | 是否只读（默认 `true`） |

```powershell
python step2_loader_and_qc.py `
  --data-root "C:\Users\User\Desktop\Binance Data V3.0" `
  --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" `
  --run-id momentum_2025 `
  --start-date "2025-01-23T17:00:00+00:00" `
  --end-date   "2025-09-01T00:00:00+00:00" `
  --bar-freq 3min `
  --symbols BTC,ETH,SOL,XRP,DOGE,1000PEPE,SUI,ADA,TRUMP,BNB,WIF,ENA,NEIRO,LINK,FARTCOIN,PNUT,LTC,AVAX,WLD,AAVE
```
结果：`universe/universe_{run_id}.csv`（覆盖率、缺失率等统计）、`config_snapshots/.../step2_args.json` 等。

---

### 5. Step2：构建 Qlib 数据集与质检
**脚本：`step2_loader_and_qc_qlib_03.py`**

```powershell
python step2_loader_and_qc_qlib_03.py `
  --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" `
  --run-id momentum_2025
  --bar-freq 3min
```
输出：更新后的 `qlib/instruments_{run_id}.txt`、`segments_{run_id}.json`、`config_snapshots/.../step2_qlib_summary.json` 等。

---

### 6. Step3：构建因子
**脚本：`step3_factor_engine_04.py`（DSL 示例）**

可直接通过 `--dsl` 参数定义因子表达式：
```powershell
python step3_factor_engine_04.py `
  --data-root "C:\Users\User\Desktop\Binance Data V3.0" `
  --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix\momentum_2025" `
  --run-id momentum_2025 `
  --bar-freq 3min `
  --symbols BTC,ETH,SOL,XRP,DOGE,1000PEPE,SUI,ADA,TRUMP,BNB,WIF,ENA,NEIRO,LINK,FARTCOIN,PNUT,LTC,AVAX,WLD,AAVE `
  --start "2025-01-23T17:00:00+00:00" `
  --end   "2025-09-01T00:00:00+00:00" `
  --dsl "MOM3_SIMPLE = close / shift(close, 3) - 1"
```
> 可替换为其他表达式（只要使用允许的字段/函数）。脚本会生成 `factors/<factor_name>/` 与 `qlib/features/<factor_name>/`。

若已有因子脚本或外部数据，也可手动写入同样的目录结构。

---

### 7. Step4：Qlib 因子评估与报告
**脚本：`step4_factor_evaluation_qlib_05.py`**

```powershell
python step4_factor_evaluation_qlib_05.py `
  --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" `
  --factor-name MOM3_SIMPLE `
  --run-id momentum_2025 `
  --bar-freq 3min `
  --horizons 5,15,30 `
  --group 5 `
  --time-split month `
  --comparison-threshold 0.05 `
  --skip-local `
  --enable-report-analysis `
  --analysis-output-dir analysis `
  --analysis-format html
```
输出：
- `reports_qlib/<factor_name>/`：IC、分组收益、对比等数据；
- `reports_qlib/<factor_name>/analysis/`：多张 PNG/HTML 图、`analysis_<factor>_<run_id>.html`、`analysis_summary.json`；
- `config_snapshots/<run_id>/step4_qlib_summary.json`：含报告路径、指标摘要等。如未安装 Plotly/Kaleido，会自动生成简单的 Matplotlib 静态图并提示。

---

### 3min 全流程示例
以下示例演示如何在 3min 频率下执行 Step0~Step4（含 Qlib 转化）：
```powershell
python step0_prepare_inputs.py `
  --project-root "C:\Users\User\Desktop\Factor Automation V3.0" `
  --data-root    "C:\Users\User\Desktop\Binance Data V3.0" `
  --output-root  "C:\Users\User\Desktop\AI_Agent_Output_fix" `
  --run-id momentum_2025 `
  --start "2025-01-23T17:00:00+00:00" `
  --end   "2025-09-01T00:00:00+00:00" `
  --symbols BTC,ETH `
  --bar-freq 3min `
  --mode execute
python step0_prepare_inputs_qlib_01.py  --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --run-id momentum_2025 --bar-freq 3min
python step1_contract_and_golden.py      --data-root "C:\Users\User\Desktop\Binance Data V3.0" --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --symbol BTCUSDT --date 2025-01-24 --bar-freq 3min
python step2_loader_and_qc.py            --data-root "C:\Users\User\Desktop\Binance Data V3.0" --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --run-id momentum_2025 --start-date 2025-01-23 --end-date 2025-09-01 --bar-freq 3min
python step3_factor_engine_04.py         --data-root "C:\Users\User\Desktop\Binance Data V3.0" --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix\momentum_2025" --run-id momentum_2025 --symbols BTCUSDT --bar-freq 3min --start 2025-01-23 --end 2025-09-01 --dsl "MOM3_SIMPLE = close / shift(close, 3) - 1"
python step4_factor_evaluation.py        --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --factor-name MOM3_SIMPLE --run-id momentum_2025 --bar-freq 3min --horizons 5,15
```
运行完成后，可在 `logs/3min/` 中查看日志并确认没有 `bar_freq=1min` 的残留提示。

### 5 分钟多币种实战示例
以下命令展示 `run-id=backtest_5min` 覆盖 50 个符号的完整流程，输出会写到 `AI_Agent_Output_fix\backtest_5min`：

1. **Step0 + 执行**
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
   > 提醒：若某符号上市时间晚于 `--start`，该符号会标记为 `first_trade_after_window`（无黄金样本/标签），Step4 Qlib 会提示 “Label file missing”。需在实际上市日之后重新执行 Step0~Step1 才能补齐标签。

2. **Step0 → Qlib**
   ```powershell
   python step0_prepare_inputs_qlib_01.py --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --run-id backtest_5min --bar-freq 5min
   ```

3. **Step1 → Qlib**
   ```powershell
   python step1_contract_and_golden_qlib_02.py --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --run-id backtest_5min --bar-freq 5min
   ```

4. **Step2 / Step2 Qlib**
   ```powershell
   python step2_loader_and_qc.py          --data-root "C:\Users\User\Desktop\Binance Data V3.0" --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --run-id backtest_5min --start-date 2023-07-28 --end-date 2025-08-30 --symbols BTC,...,1000FLOKI --bar-freq 5min
   python step2_loader_and_qc_qlib_03.py  --output-dir "C:\Users\User\Desktop\AI_Agent_Output_fix" --run-id backtest_5min --bar-freq 5min
   ```

5. **Step3 计算 VOL_BURST_COMPLEX 因子**
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

当需要重跑因子/标签时，只需清理 `backtest_5min\qlib\5min\features\VOL_BURST_COMPLEX`（或更换新的 `run-id`）后重复 Step3/Step4 即可。
\r\n### 8. 可选：测试与验证
- 冒烟测试：
  `powershell
  python -m pytest test_qlib_integration.py::test_qlib_pipeline
  `
  *该测试会在 1min 与 3min 两种频率下运行缩减版流水线，并检查日志中是否存在残留的其他频率字符串。*
- 人工审核：打开 nalysis_<factor>_<run_id>.html 浏览图表与指标，或解析 nalysis_summary.json 获取结构化信息。

---

### 9. 常见问题
- **Step0 子任务失败**：确认 `--project-root` 指向仓库；日志若提示找不到 `step1_contract_and_golden.py`，说明路径错误。
- **Step0 Qlib 缺 manifest**：确保 Step0 已生成 `step0_manifest.json` 和 `step0_summary.json`。
- **Step2 Qlib 找不到 universe**：请先运行原始 `step2_loader_and_qc.py`，并保持时间窗口/符号一致。
- **Step4 输出为空**：检查因子目录是否生成、Plotly/Kaleido 是否安装。必要时清空 `reports_qlib/<factor>` 后重跑。
- **参数格式**：在同一次命令中保持时间格式一致（全部使用 ISO8601或全部使用毫秒）。

依照上述顺序执行即可完成从数据准备、因子生成到 Qlib 报告的全流程。遇到问题可先查看 `AI_Agent_Output_fix\<run_id>\logs\` 下的日志定位原因。


