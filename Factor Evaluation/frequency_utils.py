from pathlib import Path
import logging
import math

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BaseOffset


RESAMPLED_RELATIVE = Path("6.Binance Data Resampled")
LEGACY_RELATIVE = Path("4.Binance Data Merged") / "Data"


class InvalidBarFrequency(ValueError):
    """Raised when a bar frequency string cannot be interpreted as a fixed interval."""


@dataclass(frozen=True)
class BarFrequency:
    raw: str
    canonical: str
    offset: BaseOffset
    delta: pd.Timedelta

    @property
    def minutes(self) -> int:
        return int(self.delta.total_seconds() // 60)

    @property
    def seconds(self) -> int:
        return int(self.delta.total_seconds())

    @property
    def bars_per_day(self) -> float:
        return float(pd.Timedelta(days=1) / self.delta)

    def bars_per_day_int(self) -> int:
        value = self.bars_per_day
        rounded = round(value)
        if not math.isclose(value, rounded, rel_tol=1e-9, abs_tol=1e-9):
            raise InvalidBarFrequency(
                f"Bar frequency '{self.canonical}' does not evenly divide 1 day; computed bars_per_day={value}"
            )
        return int(rounded)

    def to_timedelta(self, steps: int = 1) -> pd.Timedelta:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        return self.delta * steps

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.canonical


def _canonical_from_minutes(total_minutes: int) -> str:
    if total_minutes <= 0:
        raise InvalidBarFrequency(f"Bar frequency must be positive, got {total_minutes} minutes")
    if total_minutes == 60:
        return "60min"
    if total_minutes % 60 == 0:
        hours = total_minutes // 60
        return f"{hours}H"
    return f"{total_minutes}min"


def parse_bar_frequency(value: str) -> BarFrequency:
    if value is None:
        raise InvalidBarFrequency("Bar frequency cannot be None")
    text = str(value).strip()
    if not text:
        raise InvalidBarFrequency("Bar frequency cannot be empty")
    try:
        offset = to_offset(text)
    except (TypeError, ValueError) as exc:
        raise InvalidBarFrequency(f"Unsupported bar frequency '{value}': {exc}") from exc
    try:
        delta = pd.Timedelta(offset)
    except ValueError as exc:  # pragma: no cover - defensive
        raise InvalidBarFrequency(f"Bar frequency '{value}' is not fixed-length: {exc}") from exc
    total_seconds = delta.total_seconds()
    if total_seconds <= 0:
        raise InvalidBarFrequency(f"Bar frequency must be positive, got delta={delta}")
    if total_seconds % 60 != 0:
        raise InvalidBarFrequency(
            f"Bar frequency '{value}' must be a multiple of 1 minute; received {delta}"
        )
    total_minutes = int(total_seconds // 60)
    canonical = _canonical_from_minutes(total_minutes)
    return BarFrequency(raw=text, canonical=canonical, offset=offset, delta=delta)


def parse_bar_frequency_or_exit(value: str, *, parameter: str = "--bar-freq") -> BarFrequency:
    try:
        return parse_bar_frequency(value)
    except InvalidBarFrequency as exc:
        raise SystemExit(f"{parameter}: {exc}") from exc


def add_bar_frequency_argument(parser, *, default: str = "1min", help_text: Optional[str] = None) -> None:
    description = help_text or f"Bar frequency (default: {default}). Must match available resampled data directories (e.g. 1min, 3min, 1H)."
    parser.add_argument("--bar-freq", default=default, help=description)


def resolve_resampled_directory(data_root: Path, bar_frequency: BarFrequency) -> Path:
    return (data_root / RESAMPLED_RELATIVE / bar_frequency.canonical).expanduser().resolve()


def resolve_legacy_directory(data_root: Path) -> Path:
    return (data_root / LEGACY_RELATIVE).expanduser().resolve()


def choose_fullversion_root(
    data_root: Path,
    bar_frequency: BarFrequency,
    *,
    logger: Optional[logging.Logger] = None,
) -> tuple[Path, bool]:
    """Return directory for Fullversion CSV files and whether legacy fallback was used."""
    resampled_dir = resolve_resampled_directory(data_root, bar_frequency)
    if resampled_dir.exists():
        return resampled_dir, False
    legacy_dir = resolve_legacy_directory(data_root)
    if legacy_dir.exists():
        if logger:
            logger.warning(
                "Resampled data for bar_freq=%s missing at %s; falling back to legacy path %s",
                bar_frequency.canonical,
                resampled_dir,
                legacy_dir,
            )
        return legacy_dir, True
    raise FileNotFoundError(
        f"Neither resampled directory ({resampled_dir}) nor legacy directory ({legacy_dir}) exists."
    )
