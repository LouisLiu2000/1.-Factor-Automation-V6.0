"""Threshold-based trading strategy leveraging factor absolute levels."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Mapping

import backtrader as bt

from .base import CrossSectionStrategy


class ThresholdStrategy(CrossSectionStrategy):
    """Enter positions when factor crosses configurable thresholds."""

    params = (
        ("factor", None),
        ("buy_threshold", 1.0),
        ("sell_threshold", -1.0),
        ("position_size", 0.1),
        ("hold_max_bars", None),
        ("allow_short", False),
        ("normalization", "none"),
        ("trend_factor", None),
        ("trend_long_min", None),
        ("trend_short_max", None),
    )

    def __init__(self):
        super().__init__()
        self._hold_counters = defaultdict(int)

    def generate_target_weights(self, candidates: Mapping[bt.LineSeries, Dict[str, float]]) -> Dict[bt.LineSeries, float]:
        factor_name = self.params.factor
        if not factor_name:
            raise ValueError("ThresholdStrategy requires 'factor' parameter.")

        normalization = str(self.params.normalization or "none").lower()
        buy_threshold = float(self.params.buy_threshold)
        sell_threshold = float(self.params.sell_threshold)
        position_size = abs(float(self.params.position_size))
        hold_max = self.params.hold_max_bars if self.params.hold_max_bars else 0
        allow_short = bool(self.params.allow_short)
        trend_factor = self.params.trend_factor
        trend_long_min = float(self.params.trend_long_min) if self.params.trend_long_min is not None else None
        trend_short_max = float(self.params.trend_short_max) if self.params.trend_short_max is not None else None

        targets: Dict[bt.LineSeries, float] = {}
        raw_values: Dict[bt.LineSeries, float] = {}

        for data, factors in candidates.items():
            value = factors.get(factor_name)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                raw_values[data] = math.nan
            else:
                raw_values[data] = float(value)

        normalized_values: Dict[bt.LineSeries, float] = dict(raw_values)
        if normalization == "cs_zscore":
            valid = [v for v in raw_values.values() if not math.isnan(v)]
            if valid:
                mean = sum(valid) / len(valid)
                variance = sum((v - mean) ** 2 for v in valid) / len(valid)
                std = math.sqrt(variance)
                if std > 1e-12:
                    normalized_values = {data: (val - mean) / std if not math.isnan(val) else math.nan for data, val in raw_values.items()}
                else:
                    normalized_values = {data: 0.0 for data in raw_values}
            else:
                normalized_values = {data: math.nan for data in raw_values}

        for data, factors in candidates.items():
            value = normalized_values.get(data, math.nan)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                targets[data] = 0.0
                self._hold_counters[data] = 0
                continue

            current_position = self.getposition(data).size
            trend_value = None
            if trend_factor:
                raw_trend = factors.get(trend_factor)
                if raw_trend is not None and not (isinstance(raw_trend, float) and math.isnan(raw_trend)):
                    trend_value = float(raw_trend)

            if value >= buy_threshold:
                target_weight = position_size
                if trend_long_min is not None:
                    if trend_value is None or trend_value < trend_long_min:
                        target_weight = 0.0
            elif allow_short and value <= sell_threshold:
                target_weight = -position_size
                if trend_short_max is not None:
                    if trend_value is None or trend_value > trend_short_max:
                        target_weight = 0.0
            else:
                target_weight = 0.0

            if target_weight == 0.0:
                self._hold_counters[data] = 0
            else:
                if current_position and math.copysign(1, current_position) == math.copysign(1, target_weight):
                    self._hold_counters[data] += 1
                else:
                    self._hold_counters[data] = 1

                if hold_max and self._hold_counters[data] > hold_max:
                    target_weight = 0.0
                    self._hold_counters[data] = 0

            targets[data] = target_weight

        return targets
