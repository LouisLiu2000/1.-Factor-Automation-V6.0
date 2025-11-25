"""Common abstractions for cross-sectional Backtrader strategies."""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import backtrader as bt


logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Simple holder for per-strategy configuration."""

    max_positions: Optional[int] = None
    rebalance_frequency: Optional[int] = 1
    commission: float = 0.0
    slippage: float = 0.0
    cash_buffer: float = 0.0


class CrossSectionStrategy(bt.Strategy):
    """Base class capturing cross-sectional common logic."""

    params = (
        ("max_positions", None),
        ("rebalance_frequency", 1),
        ("commission", 0.0),
        ("slippage", 0.0),
        ("cash_buffer", 0.0),
        ("log_rankings", True),
    )

    def __init__(self):
        self._last_rebalance = -1
        self._config = StrategyConfig(
            max_positions=self.p.max_positions,
            rebalance_frequency=self.p.rebalance_frequency,
            commission=self.p.commission,
            slippage=self.p.slippage,
            cash_buffer=self.p.cash_buffer,
        )
        self._order_records: list[dict] = []
        self._position_records: list[dict] = []

    # ------------------------------ lifecycle ------------------------------ #
    def start(self):
        logger.info(
            "Starting %s | rebalance=%s | max_positions=%s | cash_buffer=%.2f",
            self.__class__.__name__,
            self._config.rebalance_frequency,
            self._config.max_positions,
            self._config.cash_buffer,
        )

    def prenext(self):
        self.next()

    def next(self):
        if not self.datas:
            return
        if not self._should_rebalance():
            return

        snapshot = self._build_snapshot()
        tradable = {
            data: factors
            for data, (factors, flag) in snapshot.items()
            if flag and factors and all(not math.isnan(v) for v in factors.values())
        }
        if not tradable:
            logger.warning("%s | no tradable instruments on bar %s", self.__class__.__name__, self.datetime.date(0))
            return

        raw_weights = self.generate_target_weights(tradable)
        if not raw_weights:
            logger.info("%s | strategy returned empty weights; skipping rebalance", self.__class__.__name__)
            return

        constrained = self._apply_constraints(raw_weights)
        self._execute_rebalance(constrained)
        self._record_positions()
        self._last_rebalance = len(self)

    # ------------------------------ helpers ------------------------------- #
    def _should_rebalance(self) -> bool:
        frequency = self._config.rebalance_frequency or 1
        if frequency <= 1:
            return True
        if self._last_rebalance < 0:
            return True
        return (len(self) - self._last_rebalance) >= frequency

    def _build_snapshot(self) -> Dict[bt.LineSeries, tuple[Dict[str, float], bool]]:
        snapshot: Dict[bt.LineSeries, tuple[Dict[str, float], bool]] = {}
        for data in self.datas:
            factors: Dict[str, float] = {}
            for column in getattr(data, "factor_columns", []):
                line = getattr(data.lines, column, None)
                if line is None:
                    continue
                value = line[0]
                if value is None:
                    factors[column] = math.nan
                    continue
                if isinstance(value, float) and math.isnan(value):
                    factors[column] = math.nan
                else:
                    factors[column] = float(value)
            tradable_flag = True
            if hasattr(data.lines, "tradable_flag"):
                flag = data.lines.tradable_flag[0]
                if flag is None or (isinstance(flag, float) and math.isnan(flag)):
                    tradable_flag = False
                else:
                    tradable_flag = bool(flag)
            snapshot[data] = (factors, tradable_flag)
        return snapshot

    def _apply_constraints(self, weights: Mapping[bt.LineSeries, float]) -> Dict[bt.LineSeries, float]:
        if not weights:
            return {}

        constrained: Dict[bt.LineSeries, float] = dict(weights)

        max_positions = self._config.max_positions
        if max_positions and max_positions > 0:
            ordered = sorted(constrained.items(), key=lambda item: abs(item[1]), reverse=True)
            trimmed = OrderedDict(ordered[: max_positions])
            constrained = dict(trimmed)

        gross_exposure = sum(abs(w) for w in constrained.values())
        buffer = min(max(self._config.cash_buffer or 0.0, 0.0), 1.0)
        if gross_exposure > 0 and buffer > 0.0:
            target_gross = max(1.0 - buffer, 0.0)
            scale = target_gross / gross_exposure
            constrained = {data: weight * scale for data, weight in constrained.items()}

        return constrained

    def _execute_rebalance(self, weights: Mapping[bt.LineSeries, float]) -> None:
        # Cancel outstanding orders to avoid overlap.
        for data in self.datas:
            open_orders = list(self.broker.get_orders_open(data))
            for order in open_orders:
                self.cancel(order)

        # Submit target weights.
        targets = dict(weights)
        for data in self.datas:
            target = targets.get(data, 0.0)
            self.order_target_percent(data=data, target=target)

        if self.params.log_rankings:
            ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
            snapshot = ", ".join(f"{data._name}:{weight:.3f}" for data, weight in ordered)
            logger.info("%s | target weights -> %s", self.__class__.__name__, snapshot)

    def _record_positions(self) -> None:
        dt = self.datetime.datetime(0)
        equity = self.broker.getvalue()
        for data in self.datas:
            position = self.getposition(data)
            market_price = data.close[0] if len(data) else float("nan")
            market_value = position.size * market_price
            weight = market_value / equity if equity else 0.0
            self._position_records.append(
                {
                    "datetime": dt.isoformat(),
                    "symbol": data._name,
                    "size": position.size,
                    "price": market_price,
                    "market_value": market_value,
                    "weight": weight,
                }
            )

    def notify_order(self, order: bt.Order) -> None:  # pragma: no cover - event driven
        if order.status not in {order.Completed, order.Canceled, order.Margin, order.Rejected}:
            return
        dt = bt.num2date(order.executed.dt) if order.executed.dt else self.datetime.datetime(0)
        self._order_records.append(
            {
                "datetime": dt.isoformat(),
                "symbol": order.data._name if order.data else None,
                "status": order.getstatusname(),
                "size": order.executed.size,
                "price": order.executed.price,
                "value": order.executed.value,
                "commission": order.executed.comm,
            }
        )

    def export_records(self) -> dict[str, list[dict]]:
        return {"orders": list(self._order_records), "positions": list(self._position_records)}

    # ------------------------------ abstract ------------------------------ #
    def generate_target_weights(self, candidates: Mapping[bt.LineSeries, Dict[str, float]]) -> Dict[bt.LineSeries, float]:
        """Return mapping data -> target weight for the current bar."""
        raise NotImplementedError
