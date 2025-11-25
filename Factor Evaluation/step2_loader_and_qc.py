#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from frequency_utils import BarFrequency, add_bar_frequency_argument, choose_fullversion_root, parse_bar_frequency_or_exit

DEFAULT_MIN_COVERAGE = 0.8

MASTER_LIST_RELATIVE = Path("5.Master Iist") / "master_item_list.csv"

BASE_COLUMNS = ["close"]


class SecurityViolation(RuntimeError):
    pass


class SafePathManager:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.logger: Optional[logging.Logger] = None

    def set_logger(self, logger: logging.Logger) -> None:
        self.logger = logger

    def _resolve(self, target: Path | str) -> Path:
        candidate = Path(target)
        resolved = (self.root / candidate).resolve()
        if not resolved.is_relative_to(self.root):
            message = f"SECURITY_VIOLATION: attempted write to {resolved}"
            if self.logger:
                self.logger.error(message)
            raise SecurityViolation(message)
        return resolved

    def prepare(self, target: Path | str) -> Path:
        resolved = self._resolve(target)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved


def detect_parquet_engine() -> str:
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except ImportError:  # pragma: no cover
        try:
            import fastparquet  # noqa: F401
            return "fastparquet"
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Neither pyarrow nor fastparquet is installed") from exc


def safe_write_csv(manager: SafePathManager, df: pd.DataFrame, path: Path | str) -> Path:
    resolved = manager.prepare(path)
    df.to_csv(resolved, index=False)
    return resolved


def safe_write_json(manager: SafePathManager, payload: Dict[str, object], path: Path | str) -> Path:
    resolved = manager.prepare(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return resolved


def ensure_datetime(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value, tz="UTC")
    except ValueError as exc:
        raise SystemExit(f"Invalid datetime '{value}': {exc}") from exc


def parse_filename_first_ms(path: Path) -> int:
    parts = path.stem.split("_")
    for token in reversed(parts):
        if token.isdigit():
            return int(token)
    raise ValueError(f"Unexpected filename format: {path.name}")


def read_first_last_ms(path: Path) -> tuple[int, int]:
    iterator = pd.read_csv(path, usecols=["open_time"], chunksize=200000)
    first_ms: Optional[int] = None
    last_ms: Optional[int] = None
    for chunk in iterator:
        if chunk.empty:
            continue
        if first_ms is None:
            first_ms = int(chunk.iloc[0, 0])
        last_ms = int(chunk.iloc[-1, 0])
    if first_ms is None or last_ms is None:
        raise ValueError(f"Empty Fullversion file: {path}")
    return first_ms, last_ms


def expand_symbols(selection: str, fullversion_dir: Path, bar_frequency: BarFrequency, allow_suffix_mismatch: bool) -> List[str]:
    files = sorted(fullversion_dir.glob("*_Fullversion*.csv"))
    if not files:
        return []

    def suffix_matches(path: Path) -> bool:
        stem = path.stem
        if stem.endswith(f"Fullversion_{bar_frequency.canonical}"):
            return True
        if allow_suffix_mismatch and stem.endswith("Fullversion"):
            return True
        return False

    filtered = [path for path in files if suffix_matches(path)]
    if not filtered:
        if allow_suffix_mismatch:
            filtered = files
        else:
            return []

    available = {path.name.split("_")[0] for path in filtered}
    if selection.strip().upper() == "ALL":
        return sorted(available)
    symbols = [s.strip() for s in selection.split(",") if s.strip()]
    resolved: List[str] = []
    for sym in symbols:
        if sym in available:
            resolved.append(sym)
        else:
            variants = [sym.rstrip(suffix) for suffix in ("USDT", "BUSD", "USDC", "TUSD", "FDUSD") if sym.endswith(suffix)]
            for alias in variants:
                if alias in available:
                    resolved.append(alias)
                    break
            else:
                resolved.append(sym)
    return sorted(set(resolved))




def build_expected_index(start: pd.Timestamp, end: pd.Timestamp, bar_frequency: BarFrequency) -> pd.DatetimeIndex:
    if end <= start:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(start=start, end=end, freq=bar_frequency.offset, inclusive="left")




def load_interval(path: Path, start_ms: int, end_ms: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=["open_time", "close"], chunksize=500000):
        mask = (chunk["open_time"] >= start_ms) & (chunk["open_time"] < end_ms)
        if mask.any():
            frames.append(chunk.loc[mask].copy())
    if not frames:
        return pd.DataFrame(columns=["open_time", "close"])
    df = pd.concat(frames, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="open_time").set_index("open_time").sort_index()
    return df


def compute_coverage(
    symbol: str,
    fullversion_dir: Path,
    metadata: List[Dict[str, object]],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    min_coverage: float,
    logger: logging.Logger,
    bar_frequency: BarFrequency,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "symbol": symbol,
        "first_time_utc": None,
        "last_time_utc": None,
        "start_date_utc": start_dt.isoformat(),
        "end_date_utc": end_dt.isoformat(),
        "expected_rows": 0,
        "actual_rows": 0,
        "coverage": 0.0,
        "missing_rate": 1.0,
        "excluded": True,
        "reason_if_excluded": "no_fullversion_file" if not metadata else "",
        "source_file": ",".join(m["path"].name for m in metadata),
        "first_ms_from_name": metadata[0]["first_ms_from_name"] if metadata else None,
        "first_ms_from_content": metadata[0]["first_ms_from_content"] if metadata else None,
    }
    if not metadata:
        return row

    earliest = metadata[0]
    row["first_time_utc"] = earliest["first_time_utc"].isoformat()
    row["first_ms_from_name"] = earliest["first_ms_from_name"]
    row["first_ms_from_content"] = earliest["first_ms_from_content"]

    last_time = max((m["last_time_utc"] for m in metadata if m["last_time_utc"] is not None), default=None)
    if last_time is None:
        row["reason_if_excluded"] = "no_valid_rows"
        return row
    row["last_time_utc"] = last_time.isoformat()

    effective_start = max(start_dt, earliest["first_time_utc"])
    effective_end = min(end_dt, last_time + bar_frequency.delta)
    if effective_end <= effective_start:
        row["reason_if_excluded"] = "data_starts_after_end"
        return row

    expected_index = build_expected_index(effective_start, effective_end, bar_frequency)
    expected_rows = len(expected_index)
    row["expected_rows"] = expected_rows
    if expected_rows == 0:
        row["reason_if_excluded"] = "no_expected_rows"
        return row

    start_ms = int(effective_start.value // 1_000_000)
    end_ms = int(effective_end.value // 1_000_000)

    frames: List[pd.DataFrame] = []
    for meta in metadata:
        df = load_interval(meta["path"], start_ms, end_ms)
        if not df.empty:
            frames.append(df)
    if frames:
        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
        aligned = combined.reindex(expected_index)
    else:
        aligned = pd.DataFrame(index=expected_index, columns=["close"])
    actual_rows = int(aligned["close"].notna().sum()) if "close" in aligned.columns else 0
    coverage = actual_rows / expected_rows if expected_rows else 0.0
    row["actual_rows"] = actual_rows
    row["coverage"] = coverage
    row["missing_rate"] = 1.0 - coverage
    if coverage >= min_coverage:
        row["excluded"] = False
    else:
        row["excluded"] = True
        row["reason_if_excluded"] = "coverage_below_threshold"
    return row




def gather_metadata(
    symbol: str,
    fullversion_dir: Path,
    logger: logging.Logger,
    bar_frequency: BarFrequency,
    allow_suffix_mismatch: bool,
) -> List[Dict[str, object]]:
    files = sorted(fullversion_dir.glob(f"{symbol}_*_Fullversion*.csv"))

    def suffix_matches(path: Path) -> bool:
        stem = path.stem
        if stem.endswith(f"Fullversion_{bar_frequency.canonical}"):
            return True
        if allow_suffix_mismatch and stem.endswith("Fullversion"):
            return True
        return False

    filtered = [path for path in files if suffix_matches(path)]
    records: List[Dict[str, object]] = []
    for path in filtered:
        try:
            name_ms = parse_filename_first_ms(path)
            first_ms, last_ms = read_first_last_ms(path)
        except Exception as exc:
            logger.warning("Skipping %s for symbol %s: %s", path.name, symbol, exc)
            continue
        if name_ms != first_ms:
            logger.warning(
                "Symbol %s file %s first ms mismatch: name=%s content=%s; using content",
                symbol,
                path.name,
                name_ms,
                first_ms,
            )
        record = {
            "path": path,
            "first_ms_from_name": name_ms,
            "first_ms_from_content": first_ms,
            "first_time_utc": pd.to_datetime(first_ms, unit="ms", utc=True),
            "last_time_utc": pd.to_datetime(last_ms, unit="ms", utc=True),
        }
        records.append(record)
    records.sort(key=lambda item: item["first_time_utc"])
    return records




def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("step2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step2 universe selector rewrite")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbols", default="ALL")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    parser.add_argument("--strict-readonly", default="true")
    parser.add_argument("--run-id")
    add_bar_frequency_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bar_frequency = parse_bar_frequency_or_exit(args.bar_freq)
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manager = SafePathManager(run_dir)
    log_path = manager.prepare(Path("logs") / bar_frequency.canonical / "step2.log")
    logger = setup_logger(log_path)
    manager.set_logger(logger)

    logger.info(
        "Universe selection start: symbols=%s start=%s end=%s min_coverage=%.2f bar_freq=%s",
        args.symbols,
        args.start_date,
        args.end_date,
        float(args.min_coverage),
        bar_frequency.canonical,
    )

    fullversion_dir, using_legacy = choose_fullversion_root(data_root, bar_frequency, logger=logger)

    start_dt = ensure_datetime(args.start_date)
    end_dt = ensure_datetime(args.end_date)
    if end_dt <= start_dt:
        raise SystemExit("end-date must be greater than start-date")

    symbols = expand_symbols(args.symbols, fullversion_dir, bar_frequency, using_legacy)
    if not symbols:
        raise SystemExit("No symbols found matching selection")

    logger.info("Discovered %d symbols", len(symbols))
    results: List[Dict[str, object]] = []

    for symbol in symbols:
        metadata = gather_metadata(symbol, fullversion_dir, logger, bar_frequency, using_legacy)
        row = compute_coverage(
            symbol,
            fullversion_dir,
            metadata,
            start_dt,
            end_dt,
            float(args.min_coverage),
            logger,
            bar_frequency,
        )
        results.append(row)
        logger.info(
            "symbol=%s first_ms_name=%s first_ms_content=%s coverage=%.4f excluded=%s reason=%s",
            symbol,
            row.get("first_ms_from_name"),
            row.get("first_ms_from_content"),
            row.get("coverage", 0.0),
            row.get("excluded"),
            row.get("reason_if_excluded"),
        )

    universe_df = pd.DataFrame(results)
    filtered = universe_df[(universe_df["excluded"] == False)]  # noqa: E712
    logger.info("Universe contains %d rows (%d included)", len(universe_df), len(filtered))

    engine = detect_parquet_engine()  # ensure dependency availability
    universe_csv = safe_write_csv(manager, universe_df, Path("universe") / bar_frequency.canonical / f"universe_{run_id}.csv")
    logger.info("Universe CSV written to %s", universe_csv)

    config_payload = {
        "run_id": run_id,
        "parameters": {
            "data_root": str(data_root),
            "output_dir": str(output_root),
            "symbols": symbols,
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "min_coverage": float(args.min_coverage),
        },
        "summary": universe_df.to_dict(orient="records"),
    }
    config_payload["parameters"]["bar_freq"] = bar_frequency.canonical
    safe_write_json(manager, config_payload, Path("config_snapshots") / bar_frequency.canonical / f"step2_{run_id}.json")
    logger.info("Configuration snapshot recorded")


if __name__ == "__main__":
    main()
