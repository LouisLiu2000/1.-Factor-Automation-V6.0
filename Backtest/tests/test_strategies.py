from collections import defaultdict
from types import SimpleNamespace

import pytest

from Backtest.strategies.longshort import LongShortStrategy
from Backtest.strategies.threshold import ThresholdStrategy
from Backtest.strategies.topk import TopKStrategy


class DummyData:
    def __init__(self, name: str) -> None:
        self._name = name


def make_params(**kwargs):
    return SimpleNamespace(**kwargs)


def test_topk_strategy_selects_top_k():
    strategy = object.__new__(TopKStrategy)
    strategy.params = make_params(factor="score", k=2, weighting="equal", min_factor=None)

    data_a = DummyData("A")
    data_b = DummyData("B")
    data_c = DummyData("C")

    candidates = {
        data_a: {"score": 5.0},
        data_b: {"score": 3.0},
        data_c: {"score": 1.0},
    }

    weights = strategy.generate_target_weights(candidates)
    assert len(weights) == 2
    assert all(abs(weight - 0.5) < 1e-9 for weight in weights.values())


def test_longshort_strategy_balances_exposure():
    strategy = object.__new__(LongShortStrategy)
    strategy.params = make_params(
        factor="alpha",
        long_quantile=0.5,
        short_quantile=0.5,
        gross_exposure=1.0,
        weighting="equal",
        hedge_ratio=1.0,
    )

    longs = DummyData("L1"), DummyData("L2")
    shorts = DummyData("S1"), DummyData("S2")
    candidates = {
        longs[0]: {"alpha": 3.0},
        longs[1]: {"alpha": 2.0},
        shorts[0]: {"alpha": -1.0},
        shorts[1]: {"alpha": -2.0},
    }

    weights = strategy.generate_target_weights(candidates)
    long_exposure = sum(weight for data, weight in weights.items() if data in longs)
    short_exposure = sum(weight for data, weight in weights.items() if data in shorts)

    assert pytest.approx(long_exposure, rel=1e-6) == 0.5
    assert pytest.approx(short_exposure, rel=1e-6) == -0.5


def test_threshold_strategy_generates_signals(monkeypatch):
    strategy = object.__new__(ThresholdStrategy)
    strategy.params = make_params(
        factor="signal",
        buy_threshold=1.0,
        sell_threshold=-1.0,
        position_size=0.2,
        hold_max_bars=None,
        allow_short=True,
    )
    strategy._hold_counters = defaultdict(int)
    strategy.getposition = lambda data: SimpleNamespace(size=0)

    data = DummyData("T1")
    weights = strategy.generate_target_weights({data: {"signal": 1.5}})
    assert pytest.approx(weights[data], rel=1e-6) == 0.2

    weights = strategy.generate_target_weights({data: {"signal": -2.0}})
    assert pytest.approx(weights[data], rel=1e-6) == -0.2
