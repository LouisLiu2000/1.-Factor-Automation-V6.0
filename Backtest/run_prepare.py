"""Stage A CLI: data preparation and factor pre-computation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from Backtest.data.dsl import DSLParseError, FactorComputationError, FactorRecipe
from Backtest.data.loader import (
    DataValidationError,
    ensure_utc_timestamp,
    load_price_frame,
    resolve_factor_directory,
    resolve_metadata_path,
    resolve_price_directory,
)
from Backtest.engine.logger import configure_logging, get_logger

LOGGER = get_logger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare factor parquet files for Backtest runs.")
    parser.add_argument("--start", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--timeframe", required=True, help="Bar timeframe (e.g. 2H, 1H).")
    parser.add_argument("--symbols", required=True, help="Comma separated list of symbols to prepare.")
    parser.add_argument("--factor-set", required=True, help="Factor set identifier to write under DataHub/factors/.")
    parser.add_argument("--dsl-path", required=True, type=Path, help="Path to DSL recipe (YAML/JSON).")
    parser.add_argument(
        "--data-root",
        default=Path("C:/Users/User/Desktop/Binance Data V3.0"),
        type=Path,
        help="Directory containing Binance resampled CSVs.",
    )
    parser.add_argument(
        "--datahub-root",
        default=Path("DataHub"),
        type=Path,
        help="Root directory for factor parquet outputs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        type=Path,
        help="Optional output root (ignored for stage A, kept for interface compatibility).",
    )
    parser.add_argument("--factor-columns", default=None, help="Unused placeholder for interface compatibility.")
    parser.add_argument("--notes", default=None, help="Optional notes to embed into metadata.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument(
        "--log-dir",
        default=Path("FactorFoundry/logs"),
        type=Path,
        help="Directory to store preparation logs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing factor parquet files if they already exist.",
    )
    return parser.parse_args(argv)


def _parse_symbols(raw: str) -> list[str]:
    symbols = [item.strip() for item in raw.split(",") if item.strip()]
    if not symbols:
        raise SystemExit("At least one symbol must be provided via --symbols.")
    return symbols


def _configure_logging(args: argparse.Namespace, run_id: str) -> None:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_dir / f"{run_id}.log"
    configure_logging(log_file=log_file, level=args.log_level)
    LOGGER.info("Logging initialised at %s", log_file)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _tradable_flag(frame: pd.DataFrame) -> pd.Series:
    return frame.notna().all(axis=1)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = _parse_symbols(args.symbols)
    start_ts = ensure_utc_timestamp(args.start)
    end_ts = ensure_utc_timestamp(args.end)
    if end_ts < start_ts:
        raise SystemExit("End date must be greater than or equal to start date.")
    if not args.dsl_path.exists():
        raise FileNotFoundError(f"DSL recipe not found: {args.dsl_path}")

    run_id = f"prepare_{args.factor_set}_{args.timeframe}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    _configure_logging(args, run_id)

    LOGGER.info(
        "Starting factor preparation | timeframe=%s | factor_set=%s | symbols=%s",
        args.timeframe,
        args.factor_set,
        ",".join(symbols),
    )
    recipe = FactorRecipe.from_file(args.dsl_path.resolve())

    csv_directory = resolve_price_directory(args.data_root, args.timeframe)
    factor_dir = resolve_factor_directory(args.datahub_root, args.timeframe, args.factor_set, create=True)
    metadata_path = resolve_metadata_path(args.datahub_root, args.timeframe, args.factor_set, create=True)
    dsl_digest = _hash_file(args.dsl_path)

    overall_nan_counts: dict[str, int] = {}
    overall_counts: dict[str, int] = {}
    per_symbol_rows: dict[str, int] = {}

    factor_columns: list[str] | None = None

    for symbol in symbols:
        LOGGER.info("Processing symbol %s", symbol)
        price_df = load_price_frame(symbol, args.timeframe, args.data_root)
        sliced = price_df.loc[start_ts:end_ts]
        if sliced.empty:
            raise DataValidationError(
                f"{symbol}: loaded price data is empty for range {start_ts.date()} - {end_ts.date()}."
            )
        try:
            factor_values = recipe.compute(sliced)
        except (FactorComputationError, DSLParseError) as exc:
            LOGGER.error("Factor computation failed for %s: %s", symbol, exc)
            raise
        if factor_columns is None:
            factor_columns = list(factor_values.columns)
        tradable = _tradable_flag(factor_values)
        output = factor_values.copy()
        output["tradable_flag"] = tradable.astype(bool)
        output.insert(0, "timestamp", output.index)

        per_symbol_rows[symbol] = len(output)
        for column in factor_values.columns:
            overall_nan_counts[column] = overall_nan_counts.get(column, 0) + factor_values[column].isna().sum()
            overall_counts[column] = overall_counts.get(column, 0) + len(factor_values)

        target_path = factor_dir / f"{symbol}.parquet"
        if target_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{target_path} already exists. Use a new factor_set identifier or pass --overwrite explicitly."
            )
        output.to_parquet(target_path, index=False)
        LOGGER.info("Wrote factor parquet %s (%d rows)", target_path, len(output))

    if factor_columns is None:
        raise SystemExit("Factor computation produced no columns; verify DSL definitions.")

    aggregate_quality = {
        column: (overall_nan_counts[column] / overall_counts[column]) if overall_counts[column] else 0.0
        for column in overall_counts
    }

    metadata_columns = ["timestamp"] + factor_columns + ["tradable_flag"]

    metadata_payload = {
        "factor_set": args.factor_set,
        "timeframe": args.timeframe,
        "source_data_path": str(csv_directory),
        "generation_time": datetime.utcnow().isoformat() + "Z",
        "code_version": "unversioned",
        "columns": metadata_columns,
        "start_time": start_ts.isoformat(),
        "end_time": end_ts.isoformat(),
        "symbols": symbols,
        "dsl_path": str(args.dsl_path.resolve()),
        "dsl_digest": dsl_digest,
        "data_quality": aggregate_quality,
        "rows_per_symbol": per_symbol_rows,
        "window_na_threshold": recipe.default_na_threshold,
        "total_rows": sum(per_symbol_rows.values()),
    }
    if args.notes:
        metadata_payload["notes"] = args.notes

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2)
    LOGGER.info("Metadata written to %s", metadata_path)

    LOGGER.info(
        "Preparation completed | factor_set=%s | symbols=%d | total_rows=%d",
        args.factor_set,
        len(symbols),
        sum(per_symbol_rows.values()),
    )

    summary = {
        "factor_set": args.factor_set,
        "timeframe": args.timeframe,
        "symbols": symbols,
        "rows_per_symbol": per_symbol_rows,
        "data_quality": aggregate_quality,
        "metadata_path": str(metadata_path),
        "factor_directory": str(factor_dir),
        "columns": metadata_columns,
        "total_rows": sum(per_symbol_rows.values()),
    }
    LOGGER.info("Preparation summary: %s", json.dumps(summary, default=str))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
