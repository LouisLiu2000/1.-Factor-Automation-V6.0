#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import importlib
from importlib import metadata as importlib_metadata
import numpy as np
import pandas as pd
import platform

from frequency_utils import (
    BarFrequency,
    add_bar_frequency_argument,
    parse_bar_frequency_or_exit,
    choose_fullversion_root,
    resolve_legacy_directory,
)

BASE_NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "premium_open",
    "premium_high",
    "premium_low",
    "premium_close",
]
OPTIONAL_COLUMNS = ["close_time", "premium_close_time"]
DESIRED_COLUMNS = ["open_time"] + BASE_NUMERIC_COLUMNS + OPTIONAL_COLUMNS


def expand_symbol_aliases(symbol: str) -> List[str]:
    aliases: List[str] = []
    def add(alias: str) -> None:
        if alias and alias not in aliases:
            aliases.append(alias)
    add(symbol)
    if symbol.endswith("PERP") and len(symbol) > 4:
        add(symbol[:-4])
    stable_suffixes = ["USDT", "BUSD", "USDC", "TUSD", "FDUSD"]
    for current in list(aliases):
        for suffix in stable_suffixes:
            if current.endswith(suffix) and len(current) > len(suffix):
                add(current[: -len(suffix)])
    return aliases


class SecurityViolation(RuntimeError):
    """Raised when attempting to write outside the allowed OUTPUT_DIR."""


class SafePathManager:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root.resolve()
        self.logger: Optional[logging.Logger] = None

    def set_logger(self, logger: logging.Logger) -> None:
        self.logger = logger

    def prepare_path(self, relative_path: Path | str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.output_root / candidate).resolve()
        if not resolved.is_relative_to(self.output_root):
            self._log_violation(resolved)
            raise SecurityViolation(f"Attempted write outside OUTPUT_DIR: {resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def _log_violation(self, target: Path) -> None:
        message = f"SECURITY_VIOLATION: attempted write to {target}"
        if self.logger:
            self.logger.error(message)
        else:
            print(message, file=sys.stderr)


def get_parquet_engine() -> str:
    for candidate in ("pyarrow", "fastparquet"):
        try:
            importlib.import_module(candidate)
        except ImportError:
            continue
        return candidate
    raise RuntimeError("Neither 'pyarrow' nor 'fastparquet' is installed; parquet output unavailable.")


def safe_write_parquet(manager: SafePathManager, df: pd.DataFrame, relative_path: Path | str) -> Path:
    target = manager.prepare_path(relative_path)
    engine = get_parquet_engine()
    df.to_parquet(target, engine=engine, index=True)
    return target


def safe_write_csv(manager: SafePathManager, df: pd.DataFrame, relative_path: Path | str) -> Path:
    target = manager.prepare_path(relative_path)
    df.to_csv(target, index=False)
    return target


def safe_write_text(manager: SafePathManager, relative_path: Path | str, content: str, encoding: str = "utf-8") -> Path:
    target = manager.prepare_path(relative_path)
    with open(target, "w", encoding=encoding) as handle:
        handle.write(content)
    return target


def safe_write_json(manager: SafePathManager, relative_path: Path | str, payload: Dict[str, Any]) -> Path:
    target = manager.prepare_path(relative_path)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return target


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("step1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Step-1 golden samples and contract artifacts.")
    parser.add_argument("--data-root", required=True, help="Root directory that contains Binance source data")
    parser.add_argument("--output-dir", required=True, help="Writable OUTPUT_DIR for golden artifacts")
    parser.add_argument("--symbol", required=True, help="Symbol to process, e.g. 1INCHUSDT")
    parser.add_argument("--date", required=True, help="UTC trading day, format YYYY-MM-DD")
    parser.add_argument("--horizons", required=True, help="Comma separated horizons, e.g. 5,15,35")
    add_bar_frequency_argument(parser)
    return parser.parse_args()


def parse_horizons(raw: str) -> List[int]:
    horizons: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Horizon must be positive: {value}")
        horizons.append(value)
    if not horizons:
        raise ValueError("At least one horizon must be provided")
    return sorted(set(horizons))


def _extract_first_timestamp(path: Path) -> Optional[int]:
    for part in reversed(path.stem.split('_')):
        if part.isdigit():
            return int(part)
    return None


def find_fullversion_file(
    directory: Path,
    symbol: str,
    target_day: pd.Timestamp,
    bar_frequency: Optional[BarFrequency],
    *,
    allow_suffix_mismatch: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"Fullversion directory not found: {directory}")
    aliases = expand_symbol_aliases(symbol)
    if target_day.tzinfo is None:
        target_day = target_day.tz_localize("UTC")
    else:
        target_day = target_day.tz_convert("UTC")
    target_ms = int(target_day.value // 1_000_000)
    date_token = target_day.strftime("%Y_%m_%d")
    expected_suffix = bar_frequency.canonical if bar_frequency is not None else None

    def suffix_matches(path: Path) -> bool:
        stem = path.stem
        if expected_suffix is None:
            return True
        if stem.endswith(f"Fullversion_{expected_suffix}"):
            return True
        return allow_suffix_mismatch and stem.endswith("Fullversion")

    candidates: List[Path] = []
    seen: set[Path] = set()
    for alias in aliases:
        for item in sorted(directory.glob(f"{alias}_*Fullversion*.csv")):
            if item in seen:
                continue
            if suffix_matches(item):
                candidates.append(item)
                seen.add(item)

    if logger is not None:
        freq_label = expected_suffix or "legacy"
        logger.info(
            "Discovered %d candidate Fullversion files in %s for symbol=%s freq=%s",
            len(candidates),
            directory,
            symbol,
            freq_label,
        )

    direct_matches = [path for path in candidates if f"_{date_token}_" in path.name]
    if direct_matches:
        return sorted(direct_matches)[0]

    ranked: List[Tuple[int, Path]] = []
    for item in candidates:
        first_ts = _extract_first_timestamp(item)
        if first_ts is not None:
            ranked.append((first_ts, item))

    if ranked:
        ranked.sort(key=lambda x: x[0])
        chosen: Optional[Path] = None
        for first_ms, path_candidate in ranked:
            if first_ms <= target_ms:
                chosen = path_candidate
            elif chosen is not None:
                break
        if chosen is None:
            chosen = ranked[0][1]
        return chosen

    if candidates:
        hint_names = [path.name for path in candidates[:5]]
    else:
        hint_names = []
    hint = ", ".join(hint_names)
    suffix_note = f" with suffix '{expected_suffix}'" if expected_suffix else ""
    raise FileNotFoundError(
        f"No Fullversion file matched aliases {aliases} for {symbol} on {target_day.date()}{suffix_note}. "
        f"Candidates: {hint if hint else 'none'}"
    )



def load_daily_slice(file_path: Path, start_ms: int, end_ms: int, usecols: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=500_000):
        mask = (chunk["open_time"] >= start_ms) & (chunk["open_time"] < end_ms)
        if mask.any():
            frames.append(chunk.loc[mask].copy())
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True)


def prepare_base_table(
    raw_df: pd.DataFrame,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    bar_frequency: BarFrequency,
) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("No rows available for requested day.")
    df = raw_df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
    for optional in OPTIONAL_COLUMNS:
        if optional in df.columns:
            df[optional] = pd.to_datetime(df[optional], unit="ms", utc=True, errors="coerce")
    df = df.set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    full_index = pd.date_range(start=start_ts, end=end_ts, freq=bar_frequency.offset, tz="UTC")
    df = df.reindex(full_index)
    df.index.name = "open_time"
    df["symbol"] = symbol
    for column in BASE_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan
    column_order = ["symbol"] + BASE_NUMERIC_COLUMNS
    for column in OPTIONAL_COLUMNS:
        if column in df.columns:
            column_order.append(column)
    df = df[column_order]
    return df



def compute_labels(base_df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    labels = pd.DataFrame(index=base_df.index)
    close = base_df["close"]
    for horizon in horizons:
        col_name = f"label_ret_{horizon}"
        shifted = close.shift(-horizon)
        labels[col_name] = shifted / close - 1
    return labels


def compute_ret_log(base_df: pd.DataFrame) -> pd.DataFrame:
    factor = pd.DataFrame(index=base_df.index)
    factor["RET_LOG_1"] = np.log(base_df["close"]) - np.log(base_df["close"].shift(1))
    return factor


def compute_rank_ic(factor: pd.Series, label: pd.Series) -> Tuple[float, int]:
    mask = factor.notna() & label.notna()
    sample_size = int(mask.sum())
    if sample_size < 2:
        return float("nan"), sample_size
    value = factor[mask].corr(label[mask], method="spearman")
    return float(value), sample_size


def find_missing_ranges(series: pd.Series, bar_frequency: BarFrequency) -> List[Dict[str, Any]]:
    if series.empty:
        return []
    missing: List[Dict[str, Any]] = []
    start: Optional[pd.Timestamp] = None
    count = 0
    for ts, is_na in series.isna().items():
        if is_na:
            if start is None:
                start = ts
                count = 1
            else:
                count += 1
        elif start is not None:
            end = ts - bar_frequency.delta
            missing.append({"start": start.isoformat(), "end": end.isoformat(), "bars": count})
            start = None
            count = 0
    if start is not None:
        missing.append({"start": start.isoformat(), "end": series.index[-1].isoformat(), "bars": count})
    return missing



def build_column_stats(df: pd.DataFrame) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    for column in df.columns:
        stats.append(
            {
                "name": column,
                "dtype": str(df[column].dtype),
                "na_ratio": float(df[column].isna().mean()),
                "non_null": int(df[column].notna().sum()),
            }
        )
    return stats


def format_column_table(stats: List[Dict[str, Any]]) -> str:
    header = ["| 列名 | dtype | NA比例 | 非空条数 |", "| --- | --- | --- | --- |"]
    rows = [f"| {item['name']} | {item['dtype']} | {item['na_ratio']:.2%} | {item['non_null']} |" for item in stats]
    return "\n".join(header + rows)


def format_missing_ranges(missing_ranges: List[Dict[str, Any]]) -> List[str]:
    if not missing_ranges:
        return ["- 无缺口（1m 等间隔满足）"]
    lines = []
    for entry in missing_ranges:
        lines.append(f"- {entry['start']} ~ {entry['end']} （{entry['bars']} bar）")
    return lines


def render_contract_readme(symbol: str, date_str: str, source_file: Path, summary: Dict[str, Any]) -> str:
    lines = [
        f"# 金样本交付说明 — {symbol} @ {date_str}",
        "",
        "## 来源文件",
        f"- Fullversion 路径：`{source_file}`",
        f"- 交易日窗口（UTC）：{summary['window_start']} ~ {summary['window_end']}",
        f"- 原始当日行数：{summary['source_rows']}",
        f"- 1m 重建索引行数：{summary['reindexed_rows']}",
        "",
        "## 转换步骤",
        "1. 只读加载 Fullversion，当日按 open_time（ms）过滤。",
        "2. open_time → Datetime[ns, UTC] 并设为索引，强制 1m reindex。",
        "3. 统一保留最小字段集，类型规范化（float64 / int64 / datetime64[ns, UTC]）。",
        "4. 计算 label_ret_5/15/35 与 RET_LOG_1，落盘到 OUTPUT_DIR/golden。",
        "",
        "## 缺口扫描",
    ]
    lines.extend(format_missing_ranges(summary["missing_ranges"]))
    lines.extend(
        [
            "",
            "## 字段统计",
            summary["column_table"],
            "",
            "## 标签 / 因子质量",
            f"- RET_LOG_1 有效样本率：{summary['factor_valid_rate']:.2%}",
        ]
    )
    for label, rate in summary["label_valid_rates"].items():
        lines.append(f"- {label} ??????{rate:.2%}")
    lines.append(
        f"- RankIC(RET_LOG_1 vs {summary['rank_ic_label']}??{summary['rank_ic']:.6f}???? {summary['rank_ic_samples']}"
    )
    lines.append("")
    lines.append("## 日志摘录")
    lines.append("- 详见 OUTPUT_DIR/logs/step1.log")
    return "\n".join(lines)


def generate_contract_markdown() -> str:
    return textwrap.dedent(
        """
        # Qlib 输入数据契约 v1.0

        ## 索引与维度
        - 时间索引：`open_time` → `Datetime[ns, UTC]`，严格 1m 等间隔，单调递增且唯一。
        - 证券维度：`symbol`（单币可为常量列，多币扩展使用长表或 MultiIndex）。

        ## 最小字段集（Fullversion 优先）
        | 分类 | 字段 | dtype | 说明 |
        | --- | --- | --- | --- |
        | 价格 | open, high, low, close | float64 | 原始价格，UTC 1m 频率 |
        | 成交 | volume, quote_asset_volume, number_of_trades, taker_buy_base_volume, taker_buy_quote_volume | float64 / int64 | number_of_trades 允许 int64 |
        | 溢价 | premium_open, premium_high, premium_low, premium_close | float64 | 允许缺失，按 NA 传播 |
        | 审计 | close_time, premium_close_time | datetime64[ns, UTC] | 可选保留 |

        ## 缺口与缺失策略
        - 强制对齐到 1m；补足索引但不造值，缺口处填充 NA。
        - Premium 缺失允许存在；因子计算按 NA 传播，评价阶段使用有效样本掩码。
        - 任意右移 `shift(-H)` 的标签尾部 H 条必须置为 NA。

        ## 标签定义
        - 回归标签：`label_ret_H = close.shift(-H) / close - 1`，其中 `H ∈ {5, 15, 35}`（单位：bar）。
        - 预留分类标签接口（本阶段不输出）。

        ## 示例因子
        - `RET_LOG_1 = log(close) - log(shift(close, 1))`，与索引对齐的一阶对数收益。

        ## 年化基数
        - 1m：每日 1440 bar，年度 525600 bar。
        - 5m：每日 288 bar，年度 105120 bar。
        - 15m：每日 96 bar，年度 35040 bar。

        ## 目录命名规范
        - `OUTPUT_DIR/golden/...`
        - `OUTPUT_DIR/config_snapshots/...`
        - `OUTPUT_DIR/logs/...`

        ## 安全守则
        - 所有写入必须通过 `safe_write_*` 且路径位于 `OUTPUT_DIR`。
        - 越权写入触发 `SECURITY_VIOLATION` 并记录于 `logs/step1.log`。
        """
    )


def gather_environment_snapshot(extra_packages: Iterable[str]) -> Dict[str, Any]:
    packages: Dict[str, Optional[str]] = {}
    for name in extra_packages:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    bar_frequency = parse_bar_frequency_or_exit(args.bar_freq)

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_manager = SafePathManager(output_dir)
    log_path = safe_manager.prepare_path(Path("logs") / bar_frequency.canonical / "step1.log")
    logger = setup_logger(log_path)
    safe_manager.set_logger(logger)

    logger.info("Step-1 start: symbol=%s date=%s horizons=%s bar_freq=%s", args.symbol, args.date, horizons, bar_frequency.canonical)

    try:
        target_day = pd.Timestamp(args.date, tz="UTC")
    except ValueError as exc:
        raise SystemExit(f"Invalid date: {args.date} ({exc})")

    start_ts = target_day
    bars_per_day = bar_frequency.bars_per_day_int()
    end_inclusive = start_ts + bar_frequency.to_timedelta(bars_per_day - 1)
    end_exclusive = start_ts + pd.Timedelta(days=1)
    start_ms = int(start_ts.value // 1_000_000)
    end_ms = int(end_exclusive.value // 1_000_000)

    fullversion_dir, using_legacy = choose_fullversion_root(data_root, bar_frequency, logger=logger)
    try:
        source_file = find_fullversion_file(
            fullversion_dir,
            args.symbol,
            start_ts,
            None if using_legacy else bar_frequency,
            allow_suffix_mismatch=using_legacy,
            logger=logger,
        )
        matched_dir = fullversion_dir
        matched_using_legacy = using_legacy
    except FileNotFoundError as primary_exc:
        if using_legacy:
            raise SystemExit(
                f"No Fullversion data for symbol={args.symbol} freq={bar_frequency.canonical} in directory {fullversion_dir}"
            ) from primary_exc
        legacy_dir = resolve_legacy_directory(data_root)
        if not legacy_dir.exists():
            raise SystemExit(
                f"Missing Fullversion data for symbol={args.symbol} freq={bar_frequency.canonical}; checked {fullversion_dir}"
            ) from primary_exc
        logger.warning(
            "No resampled Fullversion files found for bar_freq=%s symbol=%s in %s; attempting legacy fallback %s",
            bar_frequency.canonical,
            args.symbol,
            fullversion_dir,
            legacy_dir,
        )
        try:
            source_file = find_fullversion_file(
                legacy_dir,
                args.symbol,
                start_ts,
                None,
                allow_suffix_mismatch=True,
                logger=logger,
            )
        except FileNotFoundError as legacy_exc:
            raise SystemExit(
                f"Missing Fullversion data for symbol={args.symbol} freq={bar_frequency.canonical}; checked {fullversion_dir} and fallback {legacy_dir}"
            ) from legacy_exc
        matched_dir = legacy_dir
        matched_using_legacy = True
    else:
        matched_dir = fullversion_dir
        matched_using_legacy = using_legacy
    logger.info(
        "Fullversion directory selected: %s (legacy=%s)",
        matched_dir,
        matched_using_legacy,
    )
    logger.info("Using Fullversion file: %s", source_file)

    available_cols = pd.read_csv(source_file, nrows=0).columns.tolist()
    usecols = [col for col in DESIRED_COLUMNS if col in available_cols]
    if "open_time" not in usecols:
        raise SystemExit("Source file missing required column 'open_time'.")

    daily_df = load_daily_slice(source_file, start_ms, end_ms, usecols)
    source_rows = len(daily_df)
    logger.info("Rows loaded for %s: %d", args.date, source_rows)
    if daily_df.empty:
        raise SystemExit(f"No data found for {args.symbol} on {args.date}")

    base_df = prepare_base_table(daily_df, args.symbol, start_ts, end_inclusive, bar_frequency)
    reindexed_rows = len(base_df)
    missing_row_count = int(base_df["close"].isna().sum())

    missing_ranges = find_missing_ranges(base_df["close"], bar_frequency)
    column_stats = build_column_stats(base_df)
    column_table = format_column_table(column_stats)

    logger.info("Row summary: source=%d, reindexed=%d, missing_rows=%d", source_rows, reindexed_rows, missing_row_count)
    if missing_ranges:
        for entry in missing_ranges:
            logger.info("Gap: %s -> %s (%d bar)", entry["start"], entry["end"], entry["bars"])
    else:
        logger.info("Gap: none (%s coverage complete)", bar_frequency.canonical)
    for stat in column_stats:
        logger.info("NA ratio %s: %.4f (non-null=%d)", stat["name"], stat["na_ratio"], stat["non_null"])

    label_df = compute_labels(base_df, horizons)
    factor_df = compute_ret_log(base_df)

    label_valid_rates = {col: float(1 - label_df[col].isna().mean()) for col in label_df.columns}
    factor_valid_rate = float(1 - factor_df["RET_LOG_1"].isna().mean())

    logger.info("Factor RET_LOG_1 valid rate: %.4f", factor_valid_rate)
    for col, rate in label_valid_rates.items():
        logger.info("Label %s valid rate: %.4f", col, rate)

    ic_label_col = "label_ret_5" if "label_ret_5" in label_df.columns else label_df.columns[0]
    ic_value, ic_samples = compute_rank_ic(factor_df["RET_LOG_1"], label_df[ic_label_col])
    logger.info("RankIC(%s vs RET_LOG_1) = %s (samples=%d)", ic_label_col, ic_value, ic_samples)

    date_token = start_ts.strftime("%Y%m%d")
    golden_dir = Path("golden") / bar_frequency.canonical
    safe_write_parquet(safe_manager, base_df, golden_dir / f"base_{args.symbol}_{date_token}.parquet")
    safe_write_parquet(safe_manager, label_df, golden_dir / f"labels_{args.symbol}_{date_token}.parquet")
    safe_write_parquet(safe_manager, factor_df, golden_dir / f"factor_RET_LOG_1_{args.symbol}_{date_token}.parquet")

    ic_df = pd.DataFrame(
        [{"factor": "RET_LOG_1", "label": ic_label_col, "rank_ic": ic_value, "sample_size": ic_samples}]
    )
    safe_write_csv(safe_manager, ic_df, golden_dir / "ic_smoketest.csv")

    contract_readme_text = render_contract_readme(
        args.symbol,
        start_ts.strftime("%Y-%m-%d"),
        source_file,
        {
            "window_start": start_ts.isoformat(),
            "window_end": end_inclusive.isoformat(),
            "source_rows": source_rows,
            "reindexed_rows": reindexed_rows,
            "missing_ranges": missing_ranges,
            "column_table": column_table,
            "label_valid_rates": label_valid_rates,
            "factor_valid_rate": factor_valid_rate,
            "rank_ic_label": ic_label_col,
            "rank_ic": ic_value,
            "rank_ic_samples": ic_samples,
        },
    )
    safe_write_text(safe_manager, golden_dir / "contract_readme.md", contract_readme_text)

    contract_markdown = generate_contract_markdown()
    contract_doc_path = Path("docs") / bar_frequency.canonical / "Qlib-Input-Data-Contract-v1.0.md"
    safe_write_text(safe_manager, contract_doc_path, contract_markdown)

    env_snapshot = gather_environment_snapshot(["pandas", "numpy", "pyarrow", "fastparquet"])
    config_payload = {
        "parameters": {
            "data_root": str(data_root),
            "output_dir": str(output_dir),
            "symbol": args.symbol,
            "date": start_ts.strftime("%Y-%m-%d"),
            "horizons": horizons,
            "bar_freq": bar_frequency.canonical,
        },
        "source_file": str(source_file),
        "row_summary": {
            "source_rows": source_rows,
            "reindexed_rows": reindexed_rows,
            "missing_rows": missing_row_count,
        },
        "missing_ranges": missing_ranges,
        "na_ratio": {col: float(base_df[col].isna().mean()) for col in base_df.columns},
        "label_valid_rates": label_valid_rates,
        "factor_valid_rate": factor_valid_rate,
        "rank_ic": {
            "factor": "RET_LOG_1",
            "label": ic_label_col,
            "value": ic_value,
            "samples": ic_samples,
        },
        "environment": env_snapshot,
    }
    safe_write_json(safe_manager, Path("config_snapshots") / bar_frequency.canonical / "step1_args.json", config_payload)

    logger.info("Artifacts written to %s", output_dir)
    logger.info("Step-1 completed successfully.")


def run_main() -> int:
    try:
        main()
        return 0
    except SecurityViolation as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except SystemExit as exc:
        raise exc
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("step1").exception("Unhandled error: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_main())
