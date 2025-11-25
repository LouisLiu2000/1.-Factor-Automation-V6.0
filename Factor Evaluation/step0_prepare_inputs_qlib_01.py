#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from frequency_utils import add_bar_frequency_argument, parse_bar_frequency_or_exit

try:
    from step0_prepare_inputs import DEFAULT_OUTPUT_ROOT as STEP0_OUTPUT_ROOT  # type: ignore
except ImportError:
    STEP0_OUTPUT_ROOT = Path(".")

STATUS_FIELDS_QLIB = [
    "completed",
    "skipped_existing",
    "missing_fullversion",
    "first_trade_after_window",
    "failed",
    "error",
    "planned",
]


def parse_args_qlib() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Qlib calendar and instrument metadata from Step0 outputs."
    )
    parser.add_argument(
        "--output-dir",
        default=str(STEP0_OUTPUT_ROOT),
        help="Root directory that contains run outputs.",
    )
    parser.add_argument("--run-id", required=True, help="Run identifier matching Step0 execution.")
    parser.add_argument("--manifest", help="Optional explicit path to step0 manifest (json or csv).")
    parser.add_argument("--summary", help="Optional explicit path to step0 summary json.")
    parser.add_argument(
        "--calendar-freq",
        default=None,
        help="Override calendar frequency (default: match --bar-freq).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing Qlib artifacts if present.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    add_bar_frequency_argument(parser)
    return parser.parse_args()


def setup_logger_qlib(log_path: Path, level: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step0_qlib")
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


def resolve_manifest_path_qlib(run_dir: Path, manifest_override: Optional[str]) -> Path:
    if manifest_override:
        candidate = Path(manifest_override).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Manifest override not found: {candidate}")
        return candidate
    json_path = run_dir / "step0_manifest.json"
    if json_path.exists():
        return json_path
    csv_path = run_dir / "step0_manifest.csv"
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"Could not locate step0 manifest under {run_dir}")


def resolve_summary_path_qlib(run_dir: Path, summary_override: Optional[str]) -> Optional[Path]:
    if summary_override:
        candidate = Path(summary_override).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Summary override not found: {candidate}")
        return candidate
    default = run_dir / "step0_summary.json"
    return default if default.exists() else None


def load_manifest_qlib(path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading Step0 manifest from %s", path)
    if path.suffix.lower() == ".json":
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw_data = pd.read_csv(path).to_dict(orient="records")
    if not isinstance(raw_data, list):
        raise ValueError(f"Unexpected manifest payload type: {type(raw_data).__name__}")
    df = pd.DataFrame(raw_data)
    if df.empty:
        return df
    if "symbol" not in df.columns or "date" not in df.columns:
        raise ValueError("Manifest is missing required 'symbol' or 'date' fields.")
    if "status" not in df.columns:
        df["status"] = "planned"
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    logger.info("Manifest rows after cleaning: %d", len(df))
    return df


def load_summary_qlib(path: Optional[Path], logger: logging.Logger) -> Optional[Dict[str, Any]]:
    if path is None:
        logger.warning("Step0 summary file not found; falling back to manifest information.")
        return None
    logger.info("Loading Step0 summary from %s", path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected summary payload type: {type(payload).__name__}")
    return payload


def ensure_utc_timestamp_qlib(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def fallback_calendar_window_qlib(manifest_df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if manifest_df.empty:
        raise ValueError("Manifest does not contain any valid rows.")
    start_ts = manifest_df["date"].min()
    end_ts = manifest_df["date"].max()
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("Manifest dates are invalid.")
    start_ts = start_ts.normalize()
    end_ts = (end_ts.normalize() + pd.Timedelta(days=1))
    return start_ts, end_ts


def determine_calendar_window_qlib(
    summary: Optional[Dict[str, Any]],
    manifest_df: pd.DataFrame,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = ensure_utc_timestamp_qlib(summary.get("start")) if summary else None
    end_ts = ensure_utc_timestamp_qlib(summary.get("end")) if summary else None
    fallback_start, fallback_end = fallback_calendar_window_qlib(manifest_df)
    start_ts = start_ts or fallback_start
    end_ts = end_ts or fallback_end
    if end_ts <= start_ts:
        raise ValueError(f"Calendar end {end_ts} must be greater than start {start_ts}.")
    return start_ts, end_ts


def build_calendar_index_qlib(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> pd.DatetimeIndex:
    offset = pd.tseries.frequencies.to_offset(freq)
    start_aligned = start.floor(freq)
    end_aligned = end.floor(freq)
    if end != end_aligned:
        end_aligned = end_aligned + offset
    return pd.date_range(start=start_aligned, end=end_aligned, freq=freq, inclusive="left")


def build_instrument_table_qlib(manifest_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for symbol, group in manifest_df.groupby("symbol"):
        record: Dict[str, Any] = {"symbol": symbol, "total_tasks": int(len(group))}
        first_date = group["date"].min()
        last_date = group["date"].max()
        record["first_date"] = first_date.strftime("%Y-%m-%d") if pd.notna(first_date) else None
        record["last_date"] = last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else None
        for field in STATUS_FIELDS_QLIB:
            record[f"status_{field}"] = int((group["status"] == field).sum())
        available = record.get("status_completed", 0) + record.get("status_skipped_existing", 0)
        record["available_tasks"] = int(available)
        record["available_ratio"] = float(available / record["total_tasks"]) if record["total_tasks"] else 0.0
        record["needs_attention"] = available < record["total_tasks"]
        records.append(record)
    if not records:
        return pd.DataFrame(columns=["symbol", "total_tasks"])
    return pd.DataFrame(records).sort_values("symbol").reset_index(drop=True)


def write_calendar_csv_qlib(calendar_index: pd.DatetimeIndex, path: Path, force: bool) -> int:
    if path.exists() and not force:
        raise FileExistsError(f"Calendar file already exists: {path}")
    calendar_df = pd.DataFrame({"datetime": calendar_index})
    calendar_df["date"] = calendar_df["datetime"].dt.strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    calendar_df.to_csv(path, index=False)
    return len(calendar_df)


def write_instruments_csv_qlib(instruments_df: pd.DataFrame, path: Path, force: bool) -> int:
    if path.exists() and not force:
        raise FileExistsError(f"Instruments file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    instruments_df.to_csv(path, index=False)
    return len(instruments_df)


def build_snapshot_payload_qlib(
    run_id: str,
    manifest_path: Path,
    summary_path: Optional[Path],
    calendar_index: pd.DatetimeIndex,
    instruments_df: pd.DataFrame,
    status_counts: Dict[str, int],
    calendar_freq: str,
    freq_token: str,
) -> Dict[str, Any]:
    calendar_start = calendar_index[0].isoformat() if len(calendar_index) else None
    calendar_end = calendar_index[-1].isoformat() if len(calendar_index) else None
    payload = {
        "run_id": run_id,
        "bar_freq": freq_token,
        "sources": {
            "manifest": str(manifest_path),
            "summary": str(summary_path) if summary_path else None,
        },
        "calendar": {
            "frequency": calendar_freq,
            "start": calendar_start,
            "end": calendar_end,
            "bars": int(len(calendar_index)),
        },
        "status_counts": status_counts,
        "instrument_rows": instruments_df.to_dict(orient="records"),
    }
    return payload


def write_snapshot_qlib(payload: Dict[str, Any], run_dir: Path, run_id: str, freq_token: str) -> Path:
    snapshot_dir = run_dir / "config_snapshots" / freq_token / run_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / "step0_qlib_summary.json"
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot_path


def main_qlib() -> int:
    args = parse_args_qlib()
    bar_frequency = parse_bar_frequency_or_exit(args.bar_freq)
    freq_token = bar_frequency.canonical
    calendar_freq = args.calendar_freq or freq_token
    setattr(args, "bar_freq_canonical", freq_token)
    output_root = Path(args.output_dir).expanduser().resolve()
    run_root = output_root / args.run_id
    if not run_root.exists():
        print(f"Run directory not found: {run_root}")
        return 1
    preferred_manifest_dir = run_root / freq_token
    use_legacy_manifest = not preferred_manifest_dir.exists()
    manifest_root = preferred_manifest_dir if not use_legacy_manifest else run_root

    preferred_qlib_dir = run_root / "qlib" / freq_token
    legacy_qlib_dir = run_root / "qlib"
    if legacy_qlib_dir.exists() and not preferred_qlib_dir.exists():
        qlib_dir = legacy_qlib_dir
        qlib_fallback = True
    else:
        qlib_dir = preferred_qlib_dir
        qlib_fallback = False
    qlib_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_root / "logs" / freq_token / f"step0_qlib_{args.run_id}.log"
    logger = setup_logger_qlib(log_path, args.log_level)
    if use_legacy_manifest:
        logger.warning(
            "Step0 directory for bar_freq=%s not found at %s; falling back to legacy path %s",
            freq_token,
            preferred_manifest_dir,
            run_root,
        )
    logger.info("Using Step0 manifest directory: %s", manifest_root)
    if qlib_fallback:
        logger.warning(
            "Qlib directory for bar_freq=%s not found at %s; using legacy path %s",
            freq_token,
            preferred_qlib_dir,
            legacy_qlib_dir,
        )
    logger.info("Qlib outputs root: %s", qlib_dir)
    if args.calendar_freq is None:
        logger.info("Calendar frequency not provided; defaulting to bar_freq=%s", calendar_freq)
    elif args.calendar_freq != freq_token:
        logger.info("Calendar frequency override requested: %s (bar_freq=%s)", calendar_freq, freq_token)
    logger.info("Generating Qlib metadata for run_id=%s bar_freq=%s", args.run_id, freq_token)
    try:
        manifest_path = resolve_manifest_path_qlib(manifest_root, args.manifest)
    except Exception as exc:
        logger.error("Failed to locate manifest: %s", exc)
        return 1
    try:
        manifest_df = load_manifest_qlib(manifest_path, logger)
    except Exception as exc:
        logger.error("Failed to load manifest: %s", exc)
        return 1
    if manifest_df.empty:
        logger.error("Manifest is empty after loading; aborting.")
        return 1
    try:
        summary_path = resolve_summary_path_qlib(manifest_root, args.summary)
    except Exception as exc:
        logger.error("Failed to resolve summary: %s", exc)
        return 1
    try:
        summary_payload = load_summary_qlib(summary_path, logger)
    except Exception as exc:
        logger.error("Failed to load summary: %s", exc)
        return 1
    try:
        start_ts, end_ts = determine_calendar_window_qlib(summary_payload, manifest_df)
    except Exception as exc:
        logger.error("Failed to determine calendar window: %s", exc)
        return 1
    try:
        calendar_index = build_calendar_index_qlib(start_ts, end_ts, calendar_freq)
    except Exception as exc:
        logger.error("Failed to build calendar index: %s", exc)
        return 1
    calendar_path = qlib_dir / f"calendar_{args.run_id}.csv"
    try:
        bars_written = write_calendar_csv_qlib(calendar_index, calendar_path, args.force)
    except Exception as exc:
        logger.error("Failed to write calendar CSV: %s", exc)
        return 1
    logger.info("Calendar written to %s (rows=%d)", calendar_path, bars_written)
    instruments_df = build_instrument_table_qlib(manifest_df)
    instruments_path = qlib_dir / f"instruments_{args.run_id}.csv"
    try:
        instrument_rows = write_instruments_csv_qlib(instruments_df, instruments_path, args.force)
    except Exception as exc:
        logger.error("Failed to write instruments CSV: %s", exc)
        return 1
    logger.info("Instrument summary written to %s (rows=%d)", instruments_path, instrument_rows)
    status_counts_series = manifest_df["status"].value_counts(dropna=False).sort_index()
    status_counts = {str(idx): int(val) for idx, val in status_counts_series.items()}
    snapshot_payload = build_snapshot_payload_qlib(
        args.run_id,
        manifest_path,
        summary_path,
        calendar_index,
        instruments_df,
        status_counts,
        calendar_freq,
        freq_token,
    )
    snapshot_path = write_snapshot_qlib(snapshot_payload, run_root, args.run_id, freq_token)
    logger.info("Snapshot recorded at %s", snapshot_path)
    logger.info("Step0 Qlib metadata generation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_qlib())


