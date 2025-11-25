import json
from pathlib import Path

import pandas as pd
import pytest

from Backtest.engine.runner import BacktestConfig, BacktestRunner


@pytest.fixture
def sample_environment(tmp_path: Path):
    timeframe = "1H"
    factor_set = "demo_set"
    symbols = ["BTCUSDT", "ETHUSDT"]

    data_root = tmp_path / "market"
    factor_root = tmp_path / "datahub"

    for symbol in symbols:
        price_dir = data_root / "6.Binance Data Resampled" / timeframe
        price_dir.mkdir(parents=True, exist_ok=True)
        price_df = pd.DataFrame(
            {
                "open_time": [
                    1609459200000,
                    1609462800000,
                    1609466400000,
                ],
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [101 + index for index in range(3)],
                "volume": [50, 60, 55],
            }
        )
        price_path = price_dir / f"{symbol}_demo_Fullversion_{timeframe}.csv"
        price_df.to_csv(price_path, index=False)

        factor_dir = factor_root / "factors" / timeframe / factor_set
        factor_dir.mkdir(parents=True, exist_ok=True)
        factor_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2021-01-01T00:00:00Z", "2021-01-01T01:00:00Z", "2021-01-01T02:00:00Z"]
                ),
                "factor_alpha": [0.2, 0.5, 0.8] if symbol == "BTCUSDT" else [0.1, 0.3, 0.4],
                "tradable_flag": [1, 1, 1],
            }
        )
        factor_path = factor_dir / f"{symbol}.parquet"
        try:
            factor_df.to_parquet(factor_path, index=False)
        except (ImportError, ValueError):
            pytest.skip("Parquet support (pyarrow/fastparquet) not available.")

    metadata_dir = factor_root / "metadata" / timeframe
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "factor_set": factor_set,
        "timeframe": timeframe,
        "columns": ["timestamp", "factor_alpha", "tradable_flag"],
        "source_data_path": str(data_root),
        "generation_time": "2025-01-01T00:00:00Z",
        "start_time": "2021-01-01T00:00:00Z",
        "end_time": "2021-01-01T02:00:00Z",
    }
    (metadata_dir / f"{factor_set}.json").write_text(json.dumps(metadata), encoding="utf-8")

    return {
        "data_root": data_root,
        "datahub_root": factor_root,
        "timeframe": timeframe,
        "factor_set": factor_set,
        "symbols": symbols,
    }


def test_runner_produces_artifacts(tmp_path: Path, sample_environment):
    results_root = tmp_path / "results"
    config = BacktestConfig(
        start="2021-01-01",
        end="2021-01-02",
        timeframe=sample_environment["timeframe"],
        factor_set=sample_environment["factor_set"],
        symbols=sample_environment["symbols"],
        factor_columns=["factor_alpha"],
        cash=10000.0,
        commission=0.0,
        slippage=0.0,
        strategy="topk",
        strategy_params={"factor": "factor_alpha", "k": 1, "weighting": "equal"},
        rebalance_frequency=1,
        max_positions=1,
        cash_buffer=0.0,
        data_root=sample_environment["data_root"],
        datahub_root=sample_environment["datahub_root"],
        results_root=results_root,
        results_tag="pytest",
    )

    runner = BacktestRunner(config)
    result = runner.run()
    output_dir = Path(result["results_path"])
    assert output_dir.exists()
    metrics_path = output_dir / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "total_return" in metrics
    assert (output_dir / "positions.csv").exists()
    assert (output_dir / "orders.csv").exists()
    assert (output_dir / "factor_metadata.json").exists()
