"""Long-short cross-sectional strategy."""

from __future__ import annotations

import math
from typing import Dict, Mapping

import backtrader as bt

from .base import CrossSectionStrategy


class LongShortStrategy(CrossSectionStrategy):
    """Go long the top quantile and short the bottom quantile of a factor."""

    params = (
        ("factor", None),
        ("long_quantile", 0.2),
        ("short_quantile", 0.2),
        ("gross_exposure", 1.0),
        ("weighting", "equal"),
        ("hedge_ratio", 1.0),
    )

    def generate_target_weights(self, candidates: Mapping[bt.LineSeries, Dict[str, float]]) -> Dict[bt.LineSeries, float]:
        factor_name = self.params.factor
        if not factor_name:
            raise ValueError("LongShortStrategy requires 'factor' parameter.")

        values = []
        for data, factors in candidates.items():
            value = factors.get(factor_name)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            values.append((data, float(value)))
        if not values:
            return {}

        ranked = sorted(values, key=lambda item: item[1], reverse=True)
        total_count = len(ranked)

        long_q = min(max(self.params.long_quantile, 0.0), 1.0)
        short_q = min(max(self.params.short_quantile, 0.0), 1.0)

        long_count = max(int(math.floor(total_count * long_q)), 0)
        short_count = max(int(math.floor(total_count * short_q)), 0)

        longs = ranked[:long_count] if long_count else []
        shorts = ranked[-short_count:] if short_count else []

        gross = max(self.params.gross_exposure, 0.0)
        hedge = max(self.params.hedge_ratio, 0.0)

        if longs:
            long_exposure = gross / (1.0 + hedge if hedge > 0 and shorts else 1.0)
        else:
            long_exposure = 0.0
        short_exposure = long_exposure * hedge if shorts else 0.0

        weighting = str(self.params.weighting).lower()
        weights: Dict[bt.LineSeries, float] = {}

        if longs:
            if weighting == "score":
                scores = [max(score, 0.0) for _, score in longs]
                total = sum(scores)
                if total <= 0:
                    ratios = [1.0 / len(longs)] * len(longs)
                else:
                    ratios = [score / total for score in scores]
            else:
                ratios = [1.0 / len(longs)] * len(longs)
            for (data, _), ratio in zip(longs, ratios, strict=False):
                weights[data] = ratio * long_exposure

        if shorts:
            if weighting == "score":
                scores = [max(-score, 0.0) for _, score in shorts]
                total = sum(scores)
                if total <= 0:
                    ratios = [1.0 / len(shorts)] * len(shorts)
                else:
                    ratios = [score / total for score in scores]
            else:
                ratios = [1.0 / len(shorts)] * len(shorts)
            for (data, _), ratio in zip(shorts, ratios, strict=False):
                weights[data] = -ratio * short_exposure

        return weights
