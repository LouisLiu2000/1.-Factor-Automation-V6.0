#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from frequency_utils import (
    BarFrequency,
    add_bar_frequency_argument,
    parse_bar_frequency_or_exit,
)

try:
    from step0_prepare_inputs import DEFAULT_OUTPUT_ROOT as STEP_OUTPUT_ROOT  # type: ignore
except ImportError:
    STEP_OUTPUT_ROOT = Path(".")


def parse_args_qlib() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Qlib instrument.txt and segment metadata from Step2 outputs."
    )
    parser.add_argument(
        "--output-dir",
        default=str(STEP_OUTPUT_ROOT),
        help="Root directory that contains run outputs.",
    )
    parser.add_argument("--run-id", required=True, help="Run identifier matching Step2 execution.")
    parser.add_argument("--universe", help="Optional explicit path to Step2 universe CSV.")
    parser.add_argument(
        "--calendar",
        help="Optional explicit path to calendar CSV produced by Step0 Qlib script.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Train split ratio (default: 0.6).")
    parser.add_argument("--valid-ratio", type=float, default=0.2, help="Validation split ratio (default: 0.2).")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio (default: 0.2).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files when set.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    add_bar_frequency_argument(parser)
    return parser.parse_args()


def setup_logger_qlib(log_path: Path, level: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step2_qlib")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def resolve_universe_path_qlib(
    run_dir: Path, run_id: str, override: Optional[str], bar_frequency: BarFrequency, logger: logging.Logger
) -> Path:
    if override:
        candidate = Path(override).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Universe override not found: {candidate}")
        return candidate
    preferred = run_dir / "universe" / bar_frequency.canonical / f"universe_{run_id}.csv"
    if preferred.exists():
        return preferred
    legacy = run_dir / "universe" / f"universe_{run_id}.csv"
    if legacy.exists():
        logger.warning("Universe file for bar_freq=%s not found at %s; falling back to legacy path %s", bar_frequency.canonical, preferred, legacy)
        return legacy
    raise FileNotFoundError(f"Universe file not found: {preferred} or {legacy}")


def resolve_calendar_path_qlib(
    qlib_dir: Path, run_id: str, override: Optional[str], bar_frequency: BarFrequency, logger: logging.Logger
) -> Optional[Path]:
    if override:
        candidate = Path(override).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Calendar override not found: {candidate}")
        return candidate
    preferred = qlib_dir / f"calendar_{run_id}.csv"
    if preferred.exists():
        return preferred
    legacy = qlib_dir.parent / f"calendar_{run_id}.csv" if qlib_dir.name == bar_frequency.canonical else qlib_dir / f"calendar_{run_id}.csv"
    if legacy.exists() and legacy != preferred:
        logger.warning("Calendar file for bar_freq=%s not found at %s; falling back to legacy path %s", bar_frequency.canonical, preferred, legacy)
        return legacy
    return None


def as_bool_qlib(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def normalize_timestamp_qlib(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def load_universe_qlib(path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading universe data from %s", path)
    df = pd.read_csv(path)
    if df.empty:
        return df
    for column in ["first_time_utc", "last_time_utc", "start_date_utc", "end_date_utc"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")
    for column in ["coverage", "missing_rate"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ["expected_rows", "actual_rows"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["symbol"] = df["symbol"].astype(str)
    return df


def load_calendar_index_qlib(path: Optional[Path], logger: logging.Logger) -> Optional[pd.DatetimeIndex]:
    if path is None:
        logger.warning("Calendar CSV not available; deriving segments from instrument coverage only.")
        return None
    logger.info("Loading calendar index from %s", path)
    df = pd.read_csv(path, parse_dates=["datetime"])
    if "datetime" not in df.columns:
        raise ValueError("Calendar file must contain a 'datetime' column.")
    index = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dropna()
    if index.empty:
        raise ValueError("Calendar file does not contain valid datetimes.")
    return pd.DatetimeIndex(index)


def build_instrument_records_qlib(df: pd.DataFrame, logger: logging.Logger) -> Tuple[List[Dict[str, Any]], List[str]]:
    if df.empty:
        return [], []
    records: List[Dict[str, Any]] = []
    excluded_symbols: List[str] = []
    for _, row in df.iterrows():
        symbol = str(row.get("symbol"))
        if not symbol or symbol.lower() == "nan":
            continue
        excluded_flag = False
        if "excluded" in row.index:
            excluded_flag = as_bool_qlib(row.get("excluded"))
        if excluded_flag:
            excluded_symbols.append(symbol)
            continue
        start_candidates: List[pd.Timestamp] = []
        if "start_date_utc" in row.index and pd.notna(row.get("start_date_utc")):
            start_candidates.append(row["start_date_utc"])
        if "first_time_utc" in row.index and pd.notna(row.get("first_time_utc")):
            start_candidates.append(row["first_time_utc"])
        end_candidates: List[pd.Timestamp] = []
        if "end_date_utc" in row.index and pd.notna(row.get("end_date_utc")):
            end_candidates.append(row["end_date_utc"])
        if "last_time_utc" in row.index and pd.notna(row.get("last_time_utc")):
            end_candidates.append(row["last_time_utc"])
        if not start_candidates or not end_candidates:
            logger.debug("Skipping symbol %s due to missing start/end timestamps.", symbol)
            continue
        start_ts = max(start_candidates)
        end_ts = min(end_candidates)
        if end_ts <= start_ts:
            logger.debug("Skipping symbol %s because end <= start (%s <= %s).", symbol, end_ts, start_ts)
            continue
        coverage_value = row.get("coverage")
        if pd.isna(coverage_value):
            coverage_value = None
        else:
            coverage_value = float(coverage_value)
        missing_rate_value = row.get("missing_rate")
        if pd.isna(missing_rate_value):
            missing_rate_value = None
        else:
            missing_rate_value = float(missing_rate_value)
        expected_rows_value = row.get("expected_rows")
        if pd.isna(expected_rows_value):
            expected_rows = None
        else:
            expected_rows = int(expected_rows_value)
        actual_rows_value = row.get("actual_rows")
        if pd.isna(actual_rows_value):
            actual_rows = None
        else:
            actual_rows = int(actual_rows_value)
        record = {
            "symbol": symbol,
            "start_datetime": start_ts.isoformat(),
            "end_datetime": end_ts.isoformat(),
            "start_date": start_ts.strftime("%Y-%m-%d"),
            "end_date": end_ts.strftime("%Y-%m-%d"),
            "coverage": coverage_value,
            "missing_rate": missing_rate_value,
            "expected_rows": expected_rows,
            "actual_rows": actual_rows,
        }
        records.append(record)
    records.sort(key=lambda item: item["symbol"])
    return records, excluded_symbols


def write_instruments_txt_qlib(records: List[Dict[str, Any]], path: Path, force: bool) -> int:
    if path.exists() and not force:
        raise FileExistsError(f"Instruments txt already exists: {path}")
    lines = [f"{rec['symbol']} {rec['start_date']} {rec['end_date']}" for rec in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def compute_segments_qlib(
    records: List[Dict[str, Any]],
    calendar_index: Optional[pd.DatetimeIndex],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    logger: logging.Logger,
) -> Dict[str, List[str]]:
    if not records:
        raise ValueError("No instrument records available to derive segments.")
    if train_ratio <= 0 or valid_ratio <= 0 or test_ratio <= 0:
        raise ValueError("Split ratios must all be positive.")
    total_ratio = train_ratio + valid_ratio + test_ratio
    norm_train = train_ratio / total_ratio
    norm_valid = valid_ratio / total_ratio
    if calendar_index is not None and len(calendar_index) >= 3:
        total_points = len(calendar_index)
        train_cut = max(1, min(total_points - 2, int(round(total_points * norm_train))))
        valid_cut = max(train_cut + 1, min(total_points - 1, int(round(total_points * (norm_train + norm_valid)))))
        train_range = [calendar_index[0], calendar_index[train_cut - 1]]
        valid_range = [calendar_index[train_cut], calendar_index[valid_cut - 1]]
        test_range = [calendar_index[valid_cut], calendar_index[-1]]
    else:
        start_values = [normalize_timestamp_qlib(rec["start_datetime"]) for rec in records]
        end_values = [normalize_timestamp_qlib(rec["end_datetime"]) for rec in records]
        timeline_start = min(start_values)
        timeline_end = max(end_values)
        if timeline_end <= timeline_start:
            raise ValueError("Invalid global time window derived from instruments.")
        total_seconds = (timeline_end - timeline_start).total_seconds()
        train_end = timeline_start + pd.Timedelta(seconds=total_seconds * norm_train)
        valid_end = train_end + pd.Timedelta(seconds=total_seconds * norm_valid)
        train_range = [timeline_start, train_end]
        valid_range = [train_end, valid_end]
        test_range = [valid_end, timeline_end]
    segments = {
        "train": [train_range[0].isoformat(), train_range[1].isoformat()],
        "valid": [valid_range[0].isoformat(), valid_range[1].isoformat()],
        "test": [test_range[0].isoformat(), test_range[1].isoformat()],
    }
    logger.info(
        "Segments derived: train=%s valid=%s test=%s",
        segments["train"],
        segments["valid"],
        segments["test"],
    )
    return segments


def write_segments_json_qlib(segments: Dict[str, List[str]], path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Segments file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")


def build_snapshot_payload_qlib(
    run_id: str,
    universe_path: Path,
    calendar_path: Optional[Path],
    instruments_txt_path: Path,
    segments_path: Path,
    records: List[Dict[str, Any]],
    excluded_symbols: List[str],
    ratios: Dict[str, float],
    segments: Dict[str, List[str]],
    freq_token: str,
) -> Dict[str, Any]:
    coverage_values = [float(rec["coverage"]) for rec in records if rec.get("coverage") is not None]
    coverage_summary: Optional[Dict[str, float]] = None
    if coverage_values:
        coverage_summary = {
            "min": float(min(coverage_values)),
            "max": float(max(coverage_values)),
            "mean": float(sum(coverage_values) / len(coverage_values)),
        }
    payload = {
        "run_id": run_id,
        "bar_freq": freq_token,
        "sources": {
            "universe": str(universe_path),
            "calendar": str(calendar_path) if calendar_path else None,
        },
        "outputs": {
            "instruments_txt": str(instruments_txt_path),
            "segments": str(segments_path),
        },
        "ratios": ratios,
        "segments": segments,
        "instrument_records": records,
        "excluded_symbols": sorted(excluded_symbols),
        "coverage_summary": coverage_summary,
    }
    return payload


def write_snapshot_qlib(
    payload: Dict[str, Any], run_dir: Path, run_id: str, freq_token: str
) -> Path:
    snapshot_dir = run_dir / "config_snapshots" / freq_token / run_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / "step2_qlib_summary.json"
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot_path


def main_qlib() -> int:
    args = parse_args_qlib()
    bar_frequency = parse_bar_frequency_or_exit(args.bar_freq)
    freq_token = bar_frequency.canonical
    setattr(args, "bar_freq_canonical", freq_token)
    run_dir = Path(args.output_dir).expanduser().resolve() / args.run_id
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return 1
    preferred_qlib_dir = run_dir / "qlib" / freq_token
    legacy_qlib_dir = run_dir / "qlib"
    if legacy_qlib_dir.exists() and not preferred_qlib_dir.exists():
        qlib_dir = legacy_qlib_dir
        qlib_fallback = True
    else:
        qlib_dir = preferred_qlib_dir
        qlib_fallback = False
    qlib_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / freq_token / f"step2_qlib_{args.run_id}.log"
    logger = setup_logger_qlib(log_path, args.log_level)
    if qlib_fallback:
        logger.warning(
            "Qlib directory for bar_freq=%s not found at %s; using legacy path %s",
            freq_token,
            preferred_qlib_dir,
            legacy_qlib_dir,
        )
    logger.info("Generating Step2 Qlib metadata for run_id=%s bar_freq=%s", args.run_id, freq_token)
    try:
        universe_path = resolve_universe_path_qlib(run_dir, args.run_id, args.universe, bar_frequency, logger)
    except Exception as exc:
        logger.error("Failed to locate universe file: %s", exc)
        return 1
    try:
        universe_df = load_universe_qlib(universe_path, logger)
    except Exception as exc:
        logger.error("Failed to load universe data: %s", exc)
        return 1
    if universe_df.empty:
        logger.error("Universe CSV is empty; cannot derive Qlib instruments.")
        return 1
    try:
        calendar_path = resolve_calendar_path_qlib(qlib_dir, args.run_id, args.calendar, bar_frequency, logger)
    except Exception as exc:
        logger.error("Failed to resolve calendar path: %s", exc)
        return 1
    try:
        calendar_index = load_calendar_index_qlib(calendar_path, logger)
    except Exception as exc:
        logger.error("Failed to load calendar index: %s", exc)
        return 1
    try:
        records, excluded_symbols = build_instrument_records_qlib(universe_df, logger)
    except Exception as exc:
        logger.error("Failed to derive instrument records: %s", exc)
        return 1
    if not records:
        logger.error("No eligible instruments detected; aborting.")
        return 1
    instruments_txt_path = qlib_dir / f"instruments_{args.run_id}.txt"
    try:
        count_written = write_instruments_txt_qlib(records, instruments_txt_path, args.force)
    except Exception as exc:
        logger.error("Failed to write Qlib instruments txt: %s", exc)
        return 1
    logger.info("Instrument mapping written to %s (rows=%d)", instruments_txt_path, count_written)
    if excluded_symbols:
        logger.info("Excluded symbols: %d", len(excluded_symbols))
    try:
        segments = compute_segments_qlib(
            records,
            calendar_index,
            args.train_ratio,
            args.valid_ratio,
            args.test_ratio,
            logger,
        )
    except Exception as exc:
        logger.error("Failed to derive segments: %s", exc)
        return 1
    segments_path = qlib_dir / f"segments_{args.run_id}.json"
    try:
        write_segments_json_qlib(segments, segments_path, args.force)
    except Exception as exc:
        logger.error("Failed to write segments file: %s", exc)
        return 1
    ratios_payload = {
        "train": float(args.train_ratio),
        "valid": float(args.valid_ratio),
        "test": float(args.test_ratio),
    }
    snapshot_payload = build_snapshot_payload_qlib(
        args.run_id,
        universe_path,
        calendar_path,
        instruments_txt_path,
        segments_path,
        records,
        excluded_symbols,
        ratios_payload,
        segments,
        freq_token,
    )
    snapshot_path = write_snapshot_qlib(snapshot_payload, run_dir, args.run_id, freq_token)
    logger.info("Snapshot recorded at %s", snapshot_path)
    logger.info("Step2 Qlib metadata generation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_qlib())
