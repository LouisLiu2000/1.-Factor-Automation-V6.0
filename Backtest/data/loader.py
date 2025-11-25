"""Market and factor data loading utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

RESAMPLED_DIR = Path("6.Binance Data Resampled")
FACTORS_DIR = Path("factors")
METADATA_DIR = Path("metadata")


class DataValidationError(RuntimeError):
    """Raised when input data fails pre-flight validation checks."""


def ensure_utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    """Normalise timestamps to timezone-aware UTC."""

    ts = pd.Timestamp(value, tz="UTC")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with stripped column names."""

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def resolve_price_directory(data_root: Path, timeframe: str) -> Path:
    directory = (Path(data_root).expanduser() / RESAMPLED_DIR / timeframe).resolve()
    if not directory.exists():
        raise FileNotFoundError(
            f"Resampled data directory missing at {directory}. Verify timeframe '{timeframe}' is available."
        )
    return directory


def resolve_factor_directory(datahub_root: Path, timeframe: str, factor_set: str, *, create: bool = False) -> Path:
    directory = (Path(datahub_root).expanduser() / FACTORS_DIR / timeframe / factor_set).resolve()
    if directory.exists():
        return directory
    if create:
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    raise FileNotFoundError(
        f"Factor parquet directory missing at {directory}. "
        f"Run factor preparation for factor_set='{factor_set}' timeframe='{timeframe}'."
    )


def resolve_metadata_path(datahub_root: Path, timeframe: str, factor_set: str, *, create: bool = False) -> Path:
    directory = (Path(datahub_root).expanduser() / METADATA_DIR / timeframe).resolve()
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{factor_set}.json"
    if not path.exists() and not create:
        raise FileNotFoundError(
            f"Metadata JSON missing for factor_set='{factor_set}' timeframe='{timeframe}' at {path}."
        )
    return path


def load_price_frame(symbol: str, timeframe: str, data_root: Path) -> pd.DataFrame:
    """Load a Fullversion CSV for the given symbol/timeframe."""

    directory = resolve_price_directory(data_root, timeframe)
    pattern = f"{symbol}_*_Fullversion_{timeframe}.csv"
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Price CSV for symbol '{symbol}' not found under {directory} using pattern '{pattern}'."
        )
    if len(matches) > 1:
        raise DataValidationError(
            f"Multiple price CSV files matched for {symbol}: {matches}. "
            "Please consolidate to a single Fullversion file."
        )
    df = pd.read_csv(matches[0])
    df = normalise_columns(df)
    required_cols = {"open_time", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise DataValidationError(f"{symbol}: price file {matches[0]} missing columns: {sorted(missing)}")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)
    duplicated = df.index.duplicated(keep="first")
    if duplicated.any():
        duplicates = df.index[duplicated]
        raise DataValidationError(
            f"{symbol}: duplicate timestamps detected in price data: {duplicates[:5].tolist()}"
        )
    return df


def load_factor_frame(
    symbol: str,
    timeframe: str,
    factor_set: str,
    datahub_root: Path,
    expected_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    directory = resolve_factor_directory(datahub_root, timeframe, factor_set)
    path = directory / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Factor parquet for symbol '{symbol}' not found at {path}. "
            "Ensure data preparation exported this symbol."
        )
    df = pd.read_parquet(path)
    df = normalise_columns(df)
    if "timestamp" not in df.columns:
        raise DataValidationError(f"{symbol}: factor file {path} missing mandatory 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    duplicated = df.index.duplicated(keep="first")
    if duplicated.any():
        duplicates = df.index[duplicated]
        raise DataValidationError(
            f"{symbol}: duplicate timestamps detected in factor data: {duplicates[:5].tolist()}"
        )
    if expected_columns:
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            raise DataValidationError(
                f"{symbol}: requested factor columns {missing} not present. "
                f"Available columns: {sorted(df.columns)}"
            )
        df = df[list(expected_columns)]
    return df


@dataclass
class DataBundle:
    """Locate and merge market + factor data for a single backtest run."""

    symbols: Sequence[str]
    start: pd.Timestamp | str
    end: pd.Timestamp | str
    timeframe: str
    factor_set: str
    factor_columns: Sequence[str] = field(default_factory=list)
    data_root: Optional[Path] = None
    datahub_root: Optional[Path] = None
    merge_how: str = "inner"

    def __post_init__(self) -> None:
        if not self.symbols:
            raise DataValidationError("At least one symbol must be provided.")
        self.start = ensure_utc_timestamp(self.start)
        self.end = ensure_utc_timestamp(self.end)
        if self.end < self.start:
            raise DataValidationError(f"End date {self.end} is before start date {self.start}.")
        self.data_root = Path(self.data_root or Path("C:/Users/User/Desktop/Binance Data V3.0")).expanduser()
        self.datahub_root = Path(self.datahub_root or Path("DataHub")).expanduser()
        self._metadata_cache: dict[str, dict] = {}

    # ------------------------------ public API ------------------------------ #
    def load(self) -> dict[str, pd.DataFrame]:
        """Return mapping symbol -> merged DataFrame (price+factors)."""

        bundle: dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            price = load_price_frame(symbol, self.timeframe, self.data_root)
            factors = load_factor_frame(
                symbol,
                self.timeframe,
                self.factor_set,
                self.datahub_root,
                expected_columns=self._requested_columns(),
            )
            merged = self._merge_frames(price, factors)
            sliced = merged.loc[self.start : self.end]
            if sliced.empty:
                raise DataValidationError(
                    f"{symbol}: merged dataset empty after slicing to {self.start.date()} - {self.end.date()}."
                )
            bundle[symbol] = sliced
        return bundle

    # ------------------------------ loading -------------------------------- #
    def _merge_frames(self, price: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        merged = price.join(factors, how=self.merge_how)
        merged = merged.sort_index()
        return merged

    # ------------------------------ metadata -------------------------------- #
    def _metadata(self) -> dict:
        if self.factor_set in self._metadata_cache:
            return self._metadata_cache[self.factor_set]
        path = resolve_metadata_path(self.datahub_root, self.timeframe, self.factor_set)
        with path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        self._metadata_cache[self.factor_set] = meta
        return meta

    def metadata(self) -> dict:
        """Public accessor returning metadata dictionary."""

        return self._metadata()

    def _metadata_columns(self) -> list[str]:
        columns = self._metadata().get("columns")
        if not columns:
            raise DataValidationError(
                f"Metadata for factor_set='{self.factor_set}' timeframe='{self.timeframe}' missing 'columns' entry."
            )
        return [c for c in columns if c not in {"timestamp", "symbol"}]

    def _requested_columns(self) -> list[str]:
        metadata_columns = self._metadata_columns()
        requested = list(self.factor_columns) or metadata_columns
        if "tradable_flag" in metadata_columns and "tradable_flag" not in requested:
            requested.append("tradable_flag")
        return requested


def discover_factor_sets(datahub_root: Path, timeframe: str) -> Iterable[str]:
    """Yield available factor-set identifiers for the provided timeframe."""

    root = Path(datahub_root).expanduser() / FACTORS_DIR / timeframe
    if not root.exists():
        return []
    names = [entry.name for entry in root.iterdir() if entry.is_dir()]
    return sorted(names)
