"""Top-K ranking strategy based on factor scores."""

from __future__ import annotations

import math
from typing import Dict, Mapping

import backtrader as bt

from .base import CrossSectionStrategy


class TopKStrategy(CrossSectionStrategy):
    """Allocate capital to the top-K symbols ranked by a specified factor."""

    params = (
        ("factor", None),
        ("k", 5),
        ("weighting", "equal"),
        ("min_factor", None),
    )

    def generate_target_weights(self, candidates: Mapping[bt.LineSeries, Dict[str, float]]) -> Dict[bt.LineSeries, float]:
        factor_name = self.params.factor
        if not factor_name:
            raise ValueError("TopKStrategy requires 'factor' parameter.")
        ranking = []
        for data, factors in candidates.items():
            value = factors.get(factor_name)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            if self.params.min_factor is not None and value < self.params.min_factor:
                continue
            ranking.append((data, float(value)))
        ranking.sort(key=lambda item: item[1], reverse=True)
        if self.params.k and self.params.k > 0:
            ranking = ranking[: self.params.k]
        if not ranking:
            return {}
        weighting = str(self.params.weighting).lower()
        if weighting == "score":
            positives = [max(score, 0.0) for _, score in ranking]
            total = sum(positives)
            if total <= 0:
                weights = {data: 1.0 / len(ranking) for data, _ in ranking}
            else:
                weights = {data: max(score, 0.0) / total for data, score in ranking}
        elif weighting == "equal":
            weights = {data: 1.0 / len(ranking) for data, _ in ranking}
        else:
            raise ValueError(f"Unsupported weighting scheme '{self.params.weighting}'.")
        return weights
