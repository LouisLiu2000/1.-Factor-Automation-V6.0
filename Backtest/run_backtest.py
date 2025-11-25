"""CLI entrypoint for executing a single backtest run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import yaml

from Backtest.data.loader import DataValidationError
from Backtest.engine.logger import configure_logging, get_logger
from Backtest.engine.runner import BacktestConfig, BacktestRunner


logger = get_logger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a single Backtrader cross-section backtest.")
    parser.add_argument("--start", required=True, help="Inclusive backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Inclusive backtest end date (YYYY-MM-DD).")
    parser.add_argument("--timeframe", required=True, help="Bar timeframe used for both price and factors (e.g. 2H).")
    parser.add_argument("--symbols", required=True, help="Comma separated list of symbols to include.")
    parser.add_argument("--factor-set", required=True, help="Factor set identifier matching DataHub metadata.")
    parser.add_argument("--strategy", required=True, choices=["topk", "longshort", "threshold"], help="Strategy name.")
    parser.add_argument(
        "--data-root",
        default=Path("C:/Users/User/Desktop/Binance Data V3.0"),
        type=Path,
        help="Directory containing Binance resampled data (stage A/B共用默认值)。",
    )
    parser.add_argument("--datahub-root", default=Path("DataHub"), type=Path, help="Root directory for factor parquet files.")
    parser.add_argument("--output-root", default=Path("Backtest/results"), type=Path, help="Directory where results will be written.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Override strategy parameter, format key=value. Can be provided multiple times.",
    )
    parser.add_argument(
        "--factor-columns",
        default="",
        help="Comma separated subset of factor columns; defaults to all declared in metadata.",
    )
    parser.add_argument("--results-tag", default=None, help="Optional manual suffix for the run identifier.")
    parser.add_argument(
        "--dsl-path",
        default=None,
        type=Path,
        help="Optional reference to the DSL recipe used during factor preparation (for logging only).",
    )
    return parser.parse_args(argv)


def parse_params(values: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"--param requires key=value, got '{raw}'")
        key, value = raw.split("=", 1)
        params[key.strip()] = value.strip()
    return params


def coerce_value(text: str) -> object:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def extract_model_value(data: object) -> float:
    if isinstance(data, dict):
        return float(data.get("value", 0.0))
    if data is None:
        return 0.0
    return float(data)


def build_config(args: argparse.Namespace) -> BacktestConfig:
    project_root = Path(__file__).resolve().parent
    config_dir = project_root / "config"

    core_cfg = load_yaml(config_dir / "core.yaml")
    strategy_cfg = load_yaml(config_dir / "strategies" / f"{args.strategy}.yaml")

    core_defaults = core_cfg.get("defaults", {})
    strategy_defaults = dict(strategy_cfg.get("defaults", {}))
    cli_params = {key: coerce_value(value) for key, value in parse_params(args.param).items()}
    strategy_defaults.update(cli_params)

    symbols = [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
    factor_columns = [col.strip() for col in args.factor_columns.split(",") if col.strip()] if args.factor_columns else []

    config = BacktestConfig(
        start=args.start,
        end=args.end,
        timeframe=args.timeframe,
        factor_set=args.factor_set,
        symbols=symbols,
        factor_columns=factor_columns,
        cash=float(core_defaults.get("cash", 1_000_000.0)),
        commission=extract_model_value(core_defaults.get("commission")),
        slippage=extract_model_value(core_defaults.get("slippage")),
        strategy=args.strategy,
        strategy_params=strategy_defaults,
        rebalance_frequency=int(core_defaults.get("rebalance_frequency", 1) or 1),
        max_positions=core_defaults.get("max_positions"),
        cash_buffer=float(core_defaults.get("cash_buffer", 0.0)),
        data_root=args.data_root,
        datahub_root=args.datahub_root,
        results_root=args.output_root,
        results_tag=args.results_tag,
        log_level=args.log_level,
        log_rankings=bool(core_defaults.get("log_rankings", True)),
        dsl_path=args.dsl_path,
    )
    return config


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level=args.log_level)
    logger.info("Parsed arguments: %s", json.dumps(vars(args), default=str))
    try:
        config = build_config(args)
        runner = BacktestRunner(config)
        result = runner.run()
        logger.info("Run %s completed. Results stored at %s", result["run_id"], result["results_path"])
        logger.info("Key metrics: %s", json.dumps(result["metrics"], default=str))
        return 0
    except (FileNotFoundError, DataValidationError, ValueError) as exc:
        logger.error("Backtest failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
