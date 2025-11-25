import json
from pathlib import Path

import pandas as pd
import pytest

from Backtest.data.loader import DataBundle, discover_factor_sets


@pytest.fixture
def synthetic_data(tmp_path: Path):
    timeframe = "1H"
    factor_set = "demo_set"
    symbol = "BTCUSDT"

    data_root = tmp_path / "market"
    price_dir = data_root / "6.Binance Data Resampled" / timeframe
    price_dir.mkdir(parents=True)

    price_df = pd.DataFrame(
        {
            "open_time": [
                1609459200000,
                1609462800000,
                1609466400000,
            ],
            "open": [29000, 29100, 29200],
            "high": [29150, 29200, 29300],
            "low": [28900, 29050, 29100],
            "close": [29100, 29200, 29300],
            "volume": [100, 120, 110],
        }
    )
    price_path = price_dir / f"{symbol}_demo_Fullversion_{timeframe}.csv"
    price_df.to_csv(price_path, index=False)

    datahub_root = tmp_path / "datahub"
    factor_dir = datahub_root / "factors" / timeframe / factor_set
    metadata_dir = datahub_root / "metadata" / timeframe
    factor_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    factor_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2021-01-01T00:00:00Z", "2021-01-01T01:00:00Z", "2021-01-01T02:00:00Z"]
            ),
            "factor_momentum_24h": [0.5, 0.7, 0.9],
            "tradable_flag": [1, 1, 1],
        }
    )
    factor_path = factor_dir / f"{symbol}.parquet"
    try:
        factor_df.to_parquet(factor_path, index=False)
    except (ImportError, ValueError):
        pytest.skip("Parquet support (pyarrow/fastparquet) not available.")

    metadata = {
        "factor_set": factor_set,
        "timeframe": timeframe,
        "columns": ["timestamp", "factor_momentum_24h", "tradable_flag"],
        "source_data_path": str(price_dir),
        "generation_time": "2025-01-01T00:00:00Z",
        "start_time": "2021-01-01T00:00:00Z",
        "end_time": "2021-01-01T02:00:00Z",
    }
    (metadata_dir / f"{factor_set}.json").write_text(json.dumps(metadata), encoding="utf-8")

    return {
        "data_root": data_root,
        "datahub_root": datahub_root,
        "timeframe": timeframe,
        "factor_set": factor_set,
        "symbol": symbol,
    }


def test_data_bundle_merges_price_and_factors(synthetic_data):
    bundle = DataBundle(
        symbols=[synthetic_data["symbol"]],
        start="2021-01-01",
        end="2021-01-02",
        timeframe=synthetic_data["timeframe"],
        factor_set=synthetic_data["factor_set"],
        factor_columns=["factor_momentum_24h"],
        data_root=synthetic_data["data_root"],
        datahub_root=synthetic_data["datahub_root"],
    )
    merged = bundle.load()
    df = merged[synthetic_data["symbol"]]
    assert "factor_momentum_24h" in df.columns
    assert "tradable_flag" in df.columns
    assert df.index.tzinfo is not None
    assert pytest.approx(df["factor_momentum_24h"].iloc[0], rel=1e-6) == 0.5


def test_discover_factor_sets(synthetic_data):
    sets = discover_factor_sets(synthetic_data["datahub_root"], synthetic_data["timeframe"])
    assert synthetic_data["factor_set"] in sets
