#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from frequency_utils import add_bar_frequency_argument, parse_bar_frequency_or_exit, choose_fullversion_root
from step1_contract_and_golden import find_fullversion_file

DEFAULT_OUTPUT_ROOT = Path(r"C:\Users\User\Desktop\AI_Agent_Output_fix")
DEFAULT_DATA_ROOT = Path(r"C:\Users\User\Desktop\Binance Data V2.0")
DEFAULT_MASTER_LIST = Path(r"C:\Users\User\Desktop\Binance Data V2.0\5.Master Iist\master_item_list.csv")
DEFAULT_HORIZONS = "5,10,15,30,60"
DEFAULT_END_MS = 1_756_684_800_000  # 2025-09-01 00:00:00 UTC
STEP1_SCRIPT = Path("step1_contract_and_golden.py")


@dataclass
class Step1Task:
    symbol: str
    date: pd.Timestamp
    fullversion_path: Optional[Path]
    base_path: Path
    labels_path: Path
    factor_path: Path
    needs_run: bool
    log_path: Path
    bar_freq: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Step1 inputs before downstream pipeline")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id")
    parser.add_argument("--start", required=True, help="Start time (ms timestamp or ISO8601)")
    parser.add_argument("--end", help="End time (ms timestamp or ISO8601, default 1756684800000)")
    parser.add_argument("--symbols", default="ALL")
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS)
    parser.add_argument("--mode", choices=["plan", "execute"], default="plan")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--strict-readonly", default="false")
    parser.add_argument("--min-rank-volume", type=float)
    parser.add_argument("--max-rank-volume", type=float)
    parser.add_argument("--first-open-before")
    parser.add_argument("--first-open-after")
    parser.add_argument("--master-list", default=str(DEFAULT_MASTER_LIST))
    add_bar_frequency_argument(parser)
    return parser.parse_args()


def ensure_timestamp(value: Any) -> pd.Timestamp:
    if value is None:
        raise ValueError("Timestamp value is required")
    if isinstance(value, (int, float)):
        return pd.Timestamp(int(value), unit="ms", tz="UTC")
    text = str(value).strip()
    if text.isdigit():
        return pd.Timestamp(int(text), unit="ms", tz="UTC")
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def generate_date_range(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    dates: List[pd.Timestamp] = []
    current = start.floor("D")
    while current < end:
        dates.append(current)
        current += pd.Timedelta(days=1)
    return dates


def load_master_list(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)
    if "first_open_time_ms" in df.columns:
        df["first_open_time_ms"] = pd.to_numeric(df["first_open_time_ms"], errors="coerce")
    if "first_open_time_utc" in df.columns:
        df["first_open_time_utc"] = pd.to_datetime(df["first_open_time_utc"], utc=True, errors="coerce")
    return df


def resolve_symbols(
    symbols_arg: str,
    fullversion_dir: Path,
    master_df: Optional[pd.DataFrame],
    start_ts: pd.Timestamp,
    filters: Dict[str, Any],
    logger: logging.Logger,
    bar_frequency: BarFrequency,
    allow_suffix_mismatch: bool,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    files = sorted(fullversion_dir.glob("*_Fullversion*.csv"))
    if not files:
        raise SystemExit(f"No Fullversion files found in {fullversion_dir}")

    def suffix_matches(path: Path) -> bool:
        stem = path.stem
        if stem.endswith(f"Fullversion_{bar_frequency.canonical}"):
            return True
        if allow_suffix_mismatch and stem.endswith("Fullversion"):
            return True
        return False

    filtered_files = [path for path in files if suffix_matches(path)]
    if not filtered_files:
        if allow_suffix_mismatch:
            logger.warning(
                "No Fullversion files matched suffix '_Fullversion_%s' under %s; using unfiltered list.",
                bar_frequency.canonical,
                fullversion_dir,
            )
            filtered_files = files
        else:
            raise SystemExit(
                f"No Fullversion files for bar frequency {bar_frequency.canonical} found in {fullversion_dir}"
            )

    available = {path.name.split("_")[0] for path in filtered_files}
    if not available:
        raise SystemExit(f"No symbols discovered in {fullversion_dir} after filtering by bar frequency {bar_frequency.canonical}")

    if symbols_arg.strip().upper() == "ALL":
        symbols = sorted(available)
    else:
        requested = {s.strip() for s in symbols_arg.split(",") if s.strip()}
        missing = requested - available
        if missing:
            logger.warning("Symbols missing from Fullversion directory: %s", sorted(missing))
        symbols = sorted(requested & available)

    metadata: Dict[str, Dict[str, Any]] = {}
    if master_df is not None and not master_df.empty:
        df = master_df.copy()
        if filters.get("min_rank") is not None and "rank_avg_volume_360d" in df.columns:
            df = df[df["rank_avg_volume_360d"] >= filters["min_rank"]]
        if filters.get("max_rank") is not None and "rank_avg_volume_360d" in df.columns:
            df = df[df["rank_avg_volume_360d"] <= filters["max_rank"]]
        if filters.get("first_before") is not None and "first_open_time_utc" in df.columns:
            df = df[df["first_open_time_utc"] < filters["first_before"]]
        if filters.get("first_after") is not None and "first_open_time_utc" in df.columns:
            df = df[df["first_open_time_utc"] >= filters["first_after"]]
        allowed = set(df["symbol"].astype(str))
        symbols = [s for s in symbols if s in allowed]
        for _, row in df.iterrows():
            sym = str(row.get("symbol"))
            metadata[sym] = {
                "first_open_time_ms": row.get("first_open_time_ms"),
                "first_open_time_utc": row.get("first_open_time_utc"),
                "rank_avg_volume_360d": row.get("rank_avg_volume_360d"),
            }

    for sym in symbols:
        metadata.setdefault(
            sym,
            {
                "first_open_time_ms": None,
                "first_open_time_utc": None,
                "rank_avg_volume_360d": None,
            },
        )

    return symbols, metadata




def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("step0")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_task(
    symbol: str,
    date: pd.Timestamp,
    fullversion_dir: Path,
    run_dir: Path,
    logs_dir: Path,
    force: bool,
    bar_frequency: BarFrequency,
    allow_suffix_mismatch: bool,
) -> Step1Task:
    date_token = date.strftime("%Y%m%d")
    golden_dir = run_dir / "golden" / bar_frequency.canonical
    base_path = golden_dir / f"base_{symbol}_{date_token}.parquet"
    labels_path = golden_dir / f"labels_{symbol}_{date_token}.parquet"
    factor_path = golden_dir / f"factor_RET_LOG_1_{symbol}_{date_token}.parquet"
    try:
        fullversion_path = find_fullversion_file(
            fullversion_dir,
            symbol,
            date,
            bar_frequency,
            allow_suffix_mismatch=allow_suffix_mismatch,
        )
    except Exception:
        fullversion_path = None
    existing = all(p.exists() for p in (base_path, labels_path, factor_path))
    needs_run = fullversion_path is not None and (force or not existing)
    log_path = logs_dir / f"step0_step1_{symbol}_{date_token}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return Step1Task(
        symbol,
        date,
        fullversion_path,
        base_path,
        labels_path,
        factor_path,
        needs_run,
        log_path,
        bar_frequency.canonical,
    )



def run_step1(task: Step1Task, data_root: Path, output_dir: Path, project_root: Path, horizons: str, strict_readonly: str, bar_freq: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "symbol": task.symbol,
        "date": task.date.strftime("%Y-%m-%d"),
        "fullversion_path": str(task.fullversion_path) if task.fullversion_path else None,
        "bar_freq": bar_freq,
    }
    if task.fullversion_path is None:
        result["status"] = "missing_fullversion"
        result["exit_code"] = None
        return result
    if not task.needs_run:
        result["status"] = "skipped_existing"
        result["exit_code"] = 0
        return result

    task.log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(project_root / STEP1_SCRIPT),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--symbol",
        task.symbol,
        "--date",
        task.date.strftime("%Y-%m-%d"),
        "--horizons",
        horizons,
        "--bar-freq",
        bar_freq,
    ]

    start = time.perf_counter()
    try:
        proc = _run_subprocess(cmd, project_root)
        duration = (time.perf_counter() - start) * 1000
        task.log_path.write_text(proc["combined"], encoding="utf-8")
        result["exit_code"] = proc["returncode"]
        result["duration_ms"] = duration
        result["status"] = "completed" if proc["returncode"] == 0 else "failed"
    except Exception as exc:  # noqa: BLE001
        duration = (time.perf_counter() - start) * 1000
        task.log_path.write_text(str(exc), encoding="utf-8")
        result["exit_code"] = None
        result["duration_ms"] = duration
        result["status"] = "error"
        result["error"] = str(exc)
    return result


def _run_subprocess(cmd: Sequence[str], cwd: Path) -> Dict[str, Any]:
    import subprocess

    completed = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    combined = (completed.stdout or "") + (completed.stderr or "")
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "combined": combined,
    }


def sanitize_manifest(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for entry in entries:
        clean = entry.copy()
        clean.pop("task_index", None)
        sanitized.append(clean)
    return sanitized


def write_manifest(manifest: List[Dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(manifest).to_csv(csv_path, index=False, encoding="utf-8")


def evaluate_audit_ready(    tasks: List[Step1Task],    manifest: List[Dict[str, Any]],    run_dir: Path,    bar_frequency: BarFrequency,) -> bool:
    freq_token = bar_frequency.canonical
    contract_candidates = [
        run_dir / "docs" / freq_token / "Qlib-Input-Data-Contract-v1.0.md",
        run_dir / "Qlib-Input-Data-Contract-v1.0.md",
    ]
    contract = next((candidate for candidate in contract_candidates if candidate.exists()), None)
    if contract is None:
        return False
    ic_candidates = [
        run_dir / "golden" / freq_token / "ic_smoketest.csv",
        run_dir / "golden" / "ic_smoketest.csv",
    ]
    if not any(candidate.exists() for candidate in ic_candidates):
        return False
    if any(entry.get("status") == "first_trade_after_window" for entry in manifest):
        return False
    for entry in manifest:
        idx = entry.get("task_index")
        if idx is None:
            continue
        if entry.get("fullversion_path") is None:
            return False
        status = entry.get("status")
        if status not in {"completed", "skipped_existing"}:
            return False
        task = tasks[idx]
        if not all(p.exists() for p in (task.base_path, task.labels_path, task.factor_path)):
            return False
    return True


def build_summary(
    manifest: List[Dict[str, Any]],
    audit_ready: bool,
    run_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbols: Sequence[str],
    dates: Sequence[pd.Timestamp],
    mode: str,
) -> Dict[str, Any]:
    counts = {
        "completed": sum(1 for m in manifest if m.get("status") == "completed"),
        "skipped": sum(1 for m in manifest if m.get("status") == "skipped_existing"),
        "missing_fullversion": sum(1 for m in manifest if m.get("status") == "missing_fullversion"),
        "first_trade_after_window": sum(1 for m in manifest if m.get("status") == "first_trade_after_window"),
        "failed": sum(1 for m in manifest if m.get("status") == "failed"),
        "error": sum(1 for m in manifest if m.get("status") == "error"),
    }
    return {
        "run_id": run_id,
        "mode": mode,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": list(symbols),
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "total_tasks": len(manifest),
        "counts": counts,
        "audit_ready_flag": audit_ready,
    }


def main() -> int:
    args = parse_arguments()
    bar_frequency = parse_bar_frequency_or_exit(args.bar_freq)
    project_root = Path(args.project_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    start_ts = ensure_timestamp(args.start)
    end_input = args.end if args.end is not None else DEFAULT_END_MS
    end_ts = ensure_timestamp(end_input)
    if end_ts <= start_ts:
        raise SystemExit("--end must be greater than --start")

    run_id = args.run_id or pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
    run_dir = safe_mkdir(output_root / run_id)
    freq_dir = safe_mkdir(run_dir / bar_frequency.canonical)
    logs_dir = safe_mkdir(run_dir / "logs" / bar_frequency.canonical)

    logger = setup_logger(logs_dir / "step0.log")
    logger.info("Step0 start run_id=%s mode=%s bar_freq=%s", run_id, args.mode, bar_frequency.canonical)
    logger.info("Frequency-specific outputs root: %s", freq_dir)
    logger.info("Time window: %s -> %s", start_ts, end_ts)

    try:
        master_df = load_master_list(Path(args.master_list))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load master list: %s", exc)
        master_df = None

    filters: Dict[str, Any] = {}
    if args.min_rank_volume is not None:
        filters["min_rank"] = args.min_rank_volume
    if args.max_rank_volume is not None:
        filters["max_rank"] = args.max_rank_volume
    if args.first_open_before:
        filters["first_before"] = ensure_timestamp(args.first_open_before)
    if args.first_open_after:
        filters["first_after"] = ensure_timestamp(args.first_open_after)

    fullversion_dir, using_legacy = choose_fullversion_root(data_root, bar_frequency, logger=logger)
    symbols, symbol_metadata = resolve_symbols(
        args.symbols,
        fullversion_dir,
        master_df,
        start_ts,
        filters,
        logger,
        bar_frequency,
        using_legacy,
    )
    if not symbols:
        logger.error("No symbols matched the selection criteria")
        return 1

    dates = generate_date_range(start_ts, end_ts)
    if not dates:
        logger.error("No dates generated for the provided range")
        return 1

    logger.info("Symbols selected: %d", len(symbols))
    logger.info("Dates selected: %d", len(dates))

    start_ms = int(start_ts.value // 1_000_000)

    tasks: List[Step1Task] = []
    manifest: List[Dict[str, Any]] = []

    for symbol in symbols:
        meta = symbol_metadata.get(symbol, {})
        first_open_ms = meta.get("first_open_time_ms")
        first_open_utc = meta.get("first_open_time_utc")
        first_open_iso = None
        if isinstance(first_open_utc, pd.Timestamp) and not pd.isna(first_open_utc):
            first_open_iso = first_open_utc.isoformat()

        if first_open_ms is not None and not pd.isna(first_open_ms) and start_ms < int(first_open_ms):
            for date in dates:
                manifest.append(
                    {
                        "symbol": symbol,
                        "date": date.strftime("%Y-%m-%d"),
                        "fullversion_path": None,
                        "base_exists": False,
                        "labels_exists": False,
                        "factor_exists": False,
                        "needs_run": False,
                        "status": "first_trade_after_window",
                        "first_open_time_ms": int(first_open_ms),
                        "first_open_time_utc": first_open_iso,
                        "task_index": None,
                        "log_path": None,
                        "bar_freq": bar_frequency.canonical,
                    }
                )
            continue

        for date in dates:
            task = build_task(symbol, date, fullversion_dir, run_dir, logs_dir, args.force, bar_frequency, using_legacy)
            tasks.append(task)
            manifest.append(
                {
                    "symbol": symbol,
                    "date": date.strftime("%Y-%m-%d"),
                    "fullversion_path": str(task.fullversion_path) if task.fullversion_path else None,
                    "base_exists": task.base_path.exists(),
                    "labels_exists": task.labels_path.exists(),
                    "factor_exists": task.factor_path.exists(),
                    "needs_run": task.needs_run,
                    "status": "planned" if task.fullversion_path else "missing_fullversion",
                    "first_open_time_ms": int(first_open_ms) if first_open_ms is not None and not pd.isna(first_open_ms) else None,
                    "first_open_time_utc": first_open_iso,
                    "task_index": len(tasks) - 1,
                    "log_path": str(task.log_path) if task.log_path.exists() else None,
                    "bar_freq": task.bar_freq,
                }
            )

    json_path = freq_dir / "step0_manifest.json"
    csv_path = freq_dir / "step0_manifest.csv"

    if args.mode == "plan":
        sanitized = sanitize_manifest(manifest)
        write_manifest(sanitized, json_path, csv_path)
        audit_ready = evaluate_audit_ready(tasks, manifest, run_dir, bar_frequency)
        summary = build_summary(sanitized, audit_ready, run_id, start_ts, end_ts, symbols, dates, args.mode)
        (freq_dir / "step0_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Plan manifest written to %s", json_path)
        logger.info("Audit ready: %s", audit_ready)
        logger.info("Next: run with --mode execute to generate missing Step1 artifacts")
        return 0 if audit_ready else 1

    executable_tasks = [task for task in tasks if task.needs_run]
    logger.info("Tasks requiring execution: %d", len(executable_tasks))

    if executable_tasks:
        max_workers = max(1, args.concurrency)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_step1, task, data_root, run_dir, project_root, args.horizons, args.strict_readonly, task.bar_freq): task
                for task in executable_tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                idx = tasks.index(task)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "symbol": task.symbol,
                        "date": task.date.strftime("%Y-%m-%d"),
                        "status": "error",
                        "error": str(exc),
                        "exit_code": None,
                    }
                manifest[idx].update(result)
                logger.info("Step1 %s %s -> %s", task.symbol, task.date.strftime("%Y-%m-%d"), manifest[idx].get("status"))

    for entry in manifest:
        idx = entry.get("task_index")
        if idx is None:
            continue
        task = tasks[idx]
        if entry.get("status") == "planned" and task.fullversion_path and not task.needs_run:
            entry["status"] = "skipped_existing"
        entry["base_exists"] = task.base_path.exists()
        entry["labels_exists"] = task.labels_path.exists()
        entry["factor_exists"] = task.factor_path.exists()
        entry["log_path"] = str(task.log_path) if task.log_path.exists() else None
        entry["bar_freq"] = task.bar_freq

    sanitized = sanitize_manifest(manifest)
    write_manifest(sanitized, json_path, csv_path)

    audit_ready = evaluate_audit_ready(tasks, manifest, run_dir, bar_frequency)
    summary = build_summary(sanitized, audit_ready, run_id, start_ts, end_ts, symbols, dates, args.mode)
    summary_path = freq_dir / "step0_summary.json"
    summary["bar_freq"] = bar_frequency.canonical
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Execution manifest written to %s", json_path)
    logger.info("Summary written to %s", summary_path)
    logger.info("Audit ready: %s", audit_ready)
    if audit_ready:
        logger.info("All required Step1 artifacts present. Continue with audit_fix_and_rerun.py --run-id %s", run_id)
    else:
        logger.warning("Some artifacts are missing or tasks failed. Review manifest for details.")
    return 0 if audit_ready else 1


if __name__ == "__main__":
    sys.exit(main())

