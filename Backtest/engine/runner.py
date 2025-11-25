"""Single backtest orchestration logic."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import backtrader as bt
import pandas as pd

from ..data.feeds import FactorDataFeed
from ..data.loader import DataBundle
from ..strategies import (
    CrossSectionStrategy,
    LongShortStrategy,
    StrategyConfig,
    ThresholdStrategy,
    TopKStrategy,
)
from .logger import attach_file_handler, get_logger


logger = get_logger(__name__)


STRATEGY_REGISTRY: Mapping[str, type[CrossSectionStrategy]] = {
    "topk": TopKStrategy,
    "longshort": LongShortStrategy,
    "threshold": ThresholdStrategy,
}

PRICE_COLUMNS = {"open", "high", "low", "close", "volume", "openinterest"}


def _parse_timeframe_minutes(text: str) -> int:
    if text is None:
        return 1
    value = text.strip().lower()
    if value.endswith("min"):
        return int(value[:-3] or "1")
    if value.endswith("m"):
        return int(value[:-1] or "1")
    if value.endswith("h"):
        return int(value[:-1] or "1") * 60
    if value.endswith("d"):
        return int(value[:-1] or "1") * 1440
    raise ValueError(f"Unsupported timeframe '{text}'. Expected formats like '1min', '5min', '1H'.")


def _coerce_numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@dataclass
class BacktestConfig:
    """Runtime configuration for a single backtest run."""

    start: str
    end: str
    timeframe: str
    factor_set: str
    symbols: Iterable[str]
    factor_columns: Iterable[str] = field(default_factory=list)
    cash: float = 1_000_000.0
    commission: float = 0.0
    slippage: float = 0.0
    strategy: str = "topk"
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    rebalance_frequency: int = 1
    max_positions: Optional[int] = None
    cash_buffer: float = 0.0
    data_root: Optional[Path] = None
    datahub_root: Optional[Path] = None
    results_root: Optional[Path] = None
    results_tag: Optional[str] = None
    log_level: str = "INFO"
    risk_free_rate: float = 0.0
    log_rankings: bool = True
    dsl_path: Optional[Path] = None

    def clone_with(self, **updates: Any) -> "BacktestConfig":
        return replace(self, **updates)


class BacktestRunner:
    """Prepare Backtrader Cerebro instance for a fully configured run."""

    def __init__(
        self,
        config: BacktestConfig,
        bundle: Optional[DataBundle] = None,
    ):
        self.config = config
        self.bundle = bundle
        self._results_dir: Optional[Path] = None

    # ------------------------------ orchestration -------------------------- #
    def run(self) -> dict[str, Any]:
        bundle = self.bundle or self._build_bundle()
        dataset = bundle.load()
        run_id = self._build_run_id(bundle)
        results_dir = self._prepare_results_directory(run_id)
        self._results_dir = results_dir
        attach_file_handler(results_dir / "logs.txt", level=self.config.log_level)

        logger.info("Running backtest %s -> output %s", run_id, results_dir)
        cerebro = self._build_cerebro(dataset, bundle)
        strategies = cerebro.run()
        strategy = strategies[0]

        final_value = cerebro.broker.getvalue()
        metrics, equity_df = self._compute_metrics(strategy, final_value, bundle)
        logs = strategy.export_records()

        self._persist_outputs(results_dir, run_id, metrics, equity_df, logs, bundle)
        self._update_index_file(run_id, results_dir, metrics)

        return {
            "run_id": run_id,
            "results_path": str(results_dir),
            "metrics": metrics,
        }

    # ------------------------------ bundle helpers ------------------------- #
    def _build_bundle(self) -> DataBundle:
        symbols = list({s.strip(): s.strip() for s in self.config.symbols}.keys())
        return DataBundle(
            symbols=symbols,
            start=self.config.start,
            end=self.config.end,
            timeframe=self.config.timeframe,
            factor_set=self.config.factor_set,
            factor_columns=list(self.config.factor_columns),
            data_root=self.config.data_root,
            datahub_root=self.config.datahub_root,
        )

    # ------------------------------ cerebro -------------------------------- #
    def _build_cerebro(self, dataset: Dict[str, pd.DataFrame], bundle: DataBundle) -> bt.Cerebro:
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.broker.setcash(self.config.cash)
        if self.config.commission:
            cerebro.broker.setcommission(commission=self.config.commission)
        if self.config.slippage:
            cerebro.broker.set_slippage_perc(self.config.slippage)

        factor_cols = list(self.config.factor_columns)
        if not factor_cols:
            sample_df = next(iter(dataset.values()))
            factor_cols = [c for c in sample_df.columns if c not in PRICE_COLUMNS]

        feed_cls = FactorDataFeed.with_factors(factor_cols, timeframe=self.config.timeframe)

        for symbol, frame in dataset.items():
            data_feed = feed_cls(dataname=frame)
            cerebro.adddata(data_feed, name=symbol)

        strategy_cls = STRATEGY_REGISTRY.get(str(self.config.strategy).lower())
        if strategy_cls is None:
            raise ValueError(f"Unknown strategy '{self.config.strategy}'. Valid options: {sorted(STRATEGY_REGISTRY)}")

        strategy_kwargs = dict(self.config.strategy_params)
        if "max_positions" not in strategy_kwargs:
            strategy_kwargs["max_positions"] = self.config.max_positions
        if "rebalance_frequency" not in strategy_kwargs:
            strategy_kwargs["rebalance_frequency"] = self.config.rebalance_frequency
        if "commission" not in strategy_kwargs:
            strategy_kwargs["commission"] = self.config.commission
        if "slippage" not in strategy_kwargs:
            strategy_kwargs["slippage"] = self.config.slippage
        if "cash_buffer" not in strategy_kwargs:
            strategy_kwargs["cash_buffer"] = self.config.cash_buffer
        strategy_kwargs.setdefault("log_rankings", self.config.log_rankings)

        cerebro.addstrategy(strategy_cls, **strategy_kwargs)

        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=self.config.risk_free_rate)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addobserver(bt.observers.Value)

        return cerebro

    # ------------------------------ metrics -------------------------------- #
    def _compute_metrics(
        self,
        strategy: CrossSectionStrategy,
        final_value: float,
        bundle: DataBundle,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        timereturn = strategy.analyzers.timereturn.get_analysis()
        returns = pd.Series(timereturn).sort_index()
        if not returns.empty and not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
        equity = (returns + 1.0).cumprod() * self.config.cash
        equity_df = pd.DataFrame({"datetime": returns.index, "return": returns.values, "equity": equity.values})

        total_return = (final_value / self.config.cash) - 1.0
        minutes = _parse_timeframe_minutes(bundle.timeframe)
        periods_per_year = max(int((365 * 24 * 60) / minutes), 1)
        mean_return = returns.mean() if not returns.empty else 0.0
        annual_return = (1 + mean_return) ** periods_per_year - 1 if returns.size else total_return
        volatility = returns.std() * math.sqrt(periods_per_year) if returns.size else 0.0

        sharpe = strategy.analyzers.sharpe.get_analysis().get("sharperatio")
        drawdown_info = strategy.analyzers.drawdown.get_analysis()
        trade_info = strategy.analyzers.trades.get_analysis()

        max_drawdown = drawdown_info.get("max", {}).get("drawdown")
        max_drawdown_len = drawdown_info.get("max", {}).get("len")
        closed_trades = trade_info.get("total", {}).get("closed", 0)
        winning_trades = trade_info.get("won", {}).get("total", 0)
        win_rate = winning_trades / closed_trades if closed_trades else None

        metrics = {
            "start_cash": self.config.cash,
            "final_value": final_value,
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "max_drawdown_bars": max_drawdown_len,
            "total_trades": closed_trades,
            "win_rate": win_rate,
        }
        return metrics, equity_df

    # ------------------------------ persistence ---------------------------- #
    def _build_run_id(self, bundle: DataBundle) -> str:
        payload = {
            "strategy": self.config.strategy,
            "timeframe": self.config.timeframe,
            "factor_set": self.config.factor_set,
            "symbols": sorted(bundle.symbols),
            "start": str(bundle.start.date()),
            "end": str(bundle.end.date()),
            "params": self.config.strategy_params,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
        base = f"{self.config.strategy}_{self.config.timeframe}_{len(bundle.symbols)}_{digest}"
        if self.config.results_tag:
            return f"{base}_{self.config.results_tag}"
        return base

    def _prepare_results_directory(self, run_id: str) -> Path:
        root = Path(self.config.results_root or "Backtest/results").expanduser().resolve()
        output = root / self.config.strategy / run_id
        output.mkdir(parents=True, exist_ok=True)
        return output

    def _persist_outputs(
        self,
        directory: Path,
        run_id: str,
        metrics: Dict[str, Any],
        equity_df: pd.DataFrame,
        logs: Dict[str, list[dict]],
        bundle: DataBundle,
    ) -> None:
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        equity_df.to_csv(directory / "equity_curve.csv", index=False)

        if logs.get("positions"):
            pd.DataFrame.from_records(logs["positions"]).to_csv(directory / "positions.csv", index=False)
        if logs.get("orders"):
            pd.DataFrame.from_records(logs["orders"]).to_csv(directory / "orders.csv", index=False)

        metadata = bundle.metadata()
        metadata_file = directory / "factor_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        logger.info("Persisted factor metadata to %s", metadata_file)

        run_context = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "config": asdict(self.config),
            "bundle": {
                "symbols": list(bundle.symbols),
                "start": bundle.start.isoformat(),
                "end": bundle.end.isoformat(),
                "timeframe": bundle.timeframe,
                "factor_set": bundle.factor_set,
                "data_root": str(bundle.data_root),
                "datahub_root": str(bundle.datahub_root),
                "metadata": metadata,
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
            },
            "run_id": run_id,
        }
        (directory / "run_context.json").write_text(json.dumps(run_context, indent=2, default=str), encoding="utf-8")

    def _update_index_file(self, run_id: str, results_dir: Path, metrics: Dict[str, Any]) -> None:
        index_path = results_dir.parents[1] / "index.json"
        try:
            existing = json.loads(index_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            existing = []
        if not isinstance(existing, list):
            existing = []

        record = {
            "run_id": run_id,
            "strategy": self.config.strategy,
            "timeframe": self.config.timeframe,
            "factor_set": self.config.factor_set,
            "symbols": list(self.config.symbols),
            "results_path": str(results_dir),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_return": metrics.get("total_return"),
            "annual_return": metrics.get("annual_return"),
        }
        existing.append(record)
        index_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
