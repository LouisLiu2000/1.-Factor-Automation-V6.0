"""Batch parameter sweep execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .runner import BacktestRunner, BacktestConfig


@dataclass
class SweepJob:
    """Describe a single combination of strategy parameters to be executed."""

    run_id: str
    params: Dict[str, Any]


class ParameterSweep:
    """Coordinate multiple backtest runs and collate their outputs."""

    def __init__(self, base_config: BacktestConfig, jobs: Iterable[SweepJob]):
        self.base_config = base_config
        self.jobs = list(jobs)

    def execute(self) -> List[dict[str, Any]]:
        results: List[dict[str, Any]] = []
        for job in self.jobs:
            strategy_params = dict(self.base_config.strategy_params)
            strategy_params.update(job.params)
            config = self.base_config.clone_with(strategy_params=strategy_params, results_tag=job.run_id)
            runner = BacktestRunner(config)
            results.append(runner.run())
        return results
