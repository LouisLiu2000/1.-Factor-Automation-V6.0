#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from frequency_utils import add_bar_frequency_argument, parse_bar_frequency_or_exit

try:
    from step0_prepare_inputs import DEFAULT_OUTPUT_ROOT as STEP_OUTPUT_ROOT  # type: ignore
except ImportError:
    STEP_OUTPUT_ROOT = Path(".")

FACTOR_NAME_RET_LOG = "RET_LOG_1"


def parse_args_qlib() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Step1 golden artifacts into Qlib-ready parquet outputs."
    )
    parser.add_argument(
        "--output-dir",
        default=str(STEP_OUTPUT_ROOT),
        help="Root directory that contains run outputs.",
    )
    parser.add_argument("--run-id", required=True, help="Run identifier matching Step1 execution.")
    parser.add_argument(
        "--symbols",
        help="Comma-separated list of symbols to process; default processes all discovered symbols.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing parquet/json outputs when set.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    add_bar_frequency_argument(parser)
    return parser.parse_args()


def setup_logger_qlib(log_path: Path, level: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step1_qlib")
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


def parse_symbol_date_from_name_qlib(prefix: str, stem: str) -> Optional[Tuple[str, str]]:
    if not stem.startswith(prefix):
        return None
    body = stem[len(prefix) :]
    symbol, sep, date_token = body.rpartition("_")
    if not sep or not symbol or not date_token:
        return None
    return symbol, date_token


def discover_symbol_assets_qlib(golden_dir: Path, logger: logging.Logger) -> Dict[str, Dict[str, Dict[str, Path]]]:
    assets: Dict[str, Dict[str, Dict[str, Path]]] = defaultdict(lambda: {"base": {}, "labels": {}, "factor": {}})
    for path in golden_dir.glob("base_*.parquet"):
        parsed = parse_symbol_date_from_name_qlib("base_", path.stem)
        if not parsed:
            logger.debug("Skipping unexpected base filename: %s", path.name)
            continue
        symbol, date_token = parsed
        assets[symbol]["base"][date_token] = path
    for path in golden_dir.glob("labels_*.parquet"):
        parsed = parse_symbol_date_from_name_qlib("labels_", path.stem)
        if not parsed:
            logger.debug("Skipping unexpected labels filename: %s", path.name)
            continue
        symbol, date_token = parsed
        assets[symbol]["labels"][date_token] = path
    factor_prefix = f"factor_{FACTOR_NAME_RET_LOG}_"
    for path in golden_dir.glob(f"factor_{FACTOR_NAME_RET_LOG}_*.parquet"):
        parsed = parse_symbol_date_from_name_qlib(factor_prefix, path.stem)
        if not parsed:
            logger.debug("Skipping unexpected factor filename: %s", path.name)
            continue
        symbol, date_token = parsed
        assets[symbol]["factor"][date_token] = path
    return assets


def detect_parquet_engine_qlib() -> str:
    for candidate in ("pyarrow", "fastparquet"):
        try:
            __import__(candidate)
        except ImportError:
            continue
        return candidate
    raise RuntimeError("Neither pyarrow nor fastparquet is installed; parquet output unavailable.")


def load_and_concat_frames_qlib(files: Dict[str, Path], logger: logging.Logger) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for date_token, path in sorted(files.items()):
        logger.debug("Loading %s", path)
        df = pd.read_parquet(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def attach_multiindex_qlib(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df
    multi = pd.MultiIndex.from_arrays(
        [df.index, pd.Index([symbol] * len(df), name="instrument")],
        names=["datetime", "instrument"],
    )
    df = df.copy()
    df.index = multi
    return df


def write_parquet_qlib(df: pd.DataFrame, path: Path, engine: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Target already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine=engine)


def write_json_qlib(payload: Dict[str, Any], path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Target already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_meta_payload_qlib(
    symbol: str,
    base_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    factor_df: pd.DataFrame,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"symbol": symbol}
    if not base_df.empty:
        summary["start"] = base_df.index.get_level_values("datetime")[0].isoformat()
        summary["end"] = base_df.index.get_level_values("datetime")[-1].isoformat()
        summary["rows"] = int(len(base_df))
        summary["base_columns"] = list(base_df.columns)
        summary["base_na_ratio"] = {
            col: float(base_df[col].isna().mean()) for col in base_df.columns
        }
    else:
        summary["rows"] = 0
        summary["base_columns"] = []
        summary["base_na_ratio"] = {}
    if not labels_df.empty:
        summary["label_columns"] = list(labels_df.columns)
        summary["label_na_ratio"] = {
            col: float(labels_df[col].isna().mean()) for col in labels_df.columns
        }
    else:
        summary["label_columns"] = []
        summary["label_na_ratio"] = {}
    if not factor_df.empty:
        summary["factor_columns"] = list(factor_df.columns)
        summary["factor_valid_rate"] = {
            col: float(1 - factor_df[col].isna().mean()) for col in factor_df.columns
        }
    else:
        summary["factor_columns"] = []
        summary["factor_valid_rate"] = {}
    return summary


def process_symbol_qlib(
    symbol: str,
    buckets: Dict[str, Dict[str, Path]],
    engine: str,
    qlib_dir: Path,
    force: bool,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    base_paths = buckets.get("base", {})
    if not base_paths:
        logger.warning("Symbol %s skipped: no base files detected.", symbol)
        return {
            "symbol": symbol,
            "status": "missing_base",
        }, None
    base_raw = load_and_concat_frames_qlib(base_paths, logger)
    if base_raw.empty:
        logger.warning("Symbol %s skipped: base data empty.", symbol)
        return {
            "symbol": symbol,
            "status": "empty_base",
        }, None
    if "symbol" in base_raw.columns:
        base_raw = base_raw.drop(columns=[col for col in ["symbol"] if col in base_raw.columns])
    base_raw.index = pd.to_datetime(base_raw.index, utc=True)
    base_raw = base_raw.sort_index()
    union_index = base_raw.index

    label_paths = buckets.get("labels", {})
    labels_raw = load_and_concat_frames_qlib(label_paths, logger)
    if not labels_raw.empty:
        labels_raw.index = pd.to_datetime(labels_raw.index, utc=True)
        labels_raw = labels_raw.reindex(union_index)
    factor_paths = buckets.get("factor", {})
    factor_raw = load_and_concat_frames_qlib(factor_paths, logger)
    if not factor_raw.empty:
        factor_raw.index = pd.to_datetime(factor_raw.index, utc=True)
        factor_raw = factor_raw.reindex(union_index)
        if FACTOR_NAME_RET_LOG in factor_raw.columns:
            factor_raw.rename(columns={FACTOR_NAME_RET_LOG: f"${FACTOR_NAME_RET_LOG}"}, inplace=True)
    base_multi = attach_multiindex_qlib(base_raw, symbol)
    labels_multi = attach_multiindex_qlib(labels_raw, symbol) if not labels_raw.empty else pd.DataFrame(columns=[])
    factor_multi = attach_multiindex_qlib(factor_raw, symbol) if not factor_raw.empty else pd.DataFrame(columns=[])

    base_path = qlib_dir / "features" / "base" / f"{symbol}.parquet"
    write_parquet_qlib(base_multi, base_path, engine, force)

    labels_path = qlib_dir / "labels" / f"{symbol}.parquet"
    if not labels_multi.empty:
        write_parquet_qlib(labels_multi, labels_path, engine, force)
    else:
        if labels_path.exists() and force:
            labels_path.unlink()

    factor_path = qlib_dir / "features" / FACTOR_NAME_RET_LOG / f"{symbol}.parquet"
    if not factor_multi.empty:
        write_parquet_qlib(factor_multi, factor_path, engine, force)
    else:
        if factor_path.exists() and force:
            factor_path.unlink()

    meta_payload = build_meta_payload_qlib(symbol, base_multi, labels_multi, factor_multi)
    meta_path = qlib_dir / "meta" / f"{symbol}.json"
    write_json_qlib(meta_payload, meta_path, force)

    summary = {
        "symbol": symbol,
        "status": "ok",
        "rows": int(len(base_multi)),
        "base_columns": len(base_multi.columns),
        "label_columns": len(labels_multi.columns),
        "factor_columns": len(factor_multi.columns),
    }
    return summary, factor_multi if not factor_multi.empty else None


def build_snapshot_payload_qlib(
    run_id: str,
    symbols: List[str],
    summary_rows: List[Dict[str, Any]],
    outputs: Dict[str, str],
    freq_token: str,
) -> Dict[str, Any]:
    payload = {
        "run_id": run_id,
        "bar_freq": freq_token,
        "symbols": symbols,
        "summary": summary_rows,
        "outputs": outputs,
    }
    return payload


def write_factor_long_table_qlib(
    factor_frames: List[pd.DataFrame], engine: str, path: Path, force: bool, logger: logging.Logger
) -> None:
    if not factor_frames:
        logger.warning("No factor frames available for long table output at %s", path)
        return
    combined = pd.concat(factor_frames)
    combined = combined.sort_index()
    long_df = combined.reset_index()
    long_path = path
    write_parquet_qlib(long_df, long_path, engine, force)


def main_qlib() -> int:
    args = parse_args_qlib()
    bar_frequency = parse_bar_frequency_or_exit(args.bar_freq)
    freq_token = bar_frequency.canonical
    run_dir = Path(args.output_dir).expanduser().resolve() / args.run_id
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return 1
    preferred_golden_dir = run_dir / "golden" / freq_token
    legacy_golden_dir = run_dir / "golden"
    if preferred_golden_dir.exists():
        golden_dir = preferred_golden_dir
        golden_fallback = False
    elif legacy_golden_dir.exists():
        golden_dir = legacy_golden_dir
        golden_fallback = True
    else:
        print(f"Golden directory not found for bar_freq={freq_token}: {preferred_golden_dir}")
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
    log_path = run_dir / "logs" / freq_token / f"step1_qlib_{args.run_id}.log"
    logger = setup_logger_qlib(log_path, args.log_level)
    if golden_fallback:
        logger.warning(
            "Golden directory for bar_freq=%s not found at %s; using legacy path %s",
            freq_token,
            preferred_golden_dir,
            legacy_golden_dir,
        )
    if qlib_fallback:
        logger.warning(
            "Qlib directory for bar_freq=%s not found at %s; using legacy path %s",
            freq_token,
            preferred_qlib_dir,
            legacy_qlib_dir,
        )
    logger.info(
        "Using golden directory %s (fallback=%s) for bar_freq=%s",
        golden_dir,
        golden_fallback,
        freq_token,
    )
    logger.info(
        "Qlib output directory resolved to %s (fallback=%s)",
        qlib_dir,
        qlib_fallback,
    )
    logger.info("Converting Step1 golden data into Qlib format for run_id=%s bar_freq=%s", args.run_id, freq_token)
    assets = discover_symbol_assets_qlib(golden_dir, logger)
    if not assets:
        logger.error("No golden artifacts detected under %s", golden_dir)
        return 1
    if args.symbols:
        requested = {sym.strip() for sym in args.symbols.split(",") if sym.strip()}
        assets = {symbol: buckets for symbol, buckets in assets.items() if symbol in requested}
        missing = requested - set(assets.keys())
        if missing:
            logger.warning("Requested symbols missing from golden data: %s", sorted(missing))
    if not assets:
        logger.error("No symbols remain after filtering; aborting.")
        return 1
    try:
        engine = detect_parquet_engine_qlib()
    except Exception as exc:
        logger.error("Unable to determine parquet engine: %s", exc)
        return 1

    factor_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for symbol in sorted(assets.keys()):
        try:
            summary, factor_multi = process_symbol_qlib(symbol, assets[symbol], engine, qlib_dir, args.force, logger)
        except FileExistsError as exc:
            logger.error("%s", exc)
            return 1
        except Exception as exc:
            logger.exception("Failed processing symbol %s: %s", symbol, exc)
            summary_rows.append({"symbol": symbol, "status": "error", "error": str(exc)})
            continue
        summary_rows.append(summary)
        if factor_multi is not None and not factor_multi.empty:
            factor_frames.append(factor_multi)

    factor_long_path = qlib_dir / "features" / f"{FACTOR_NAME_RET_LOG}_long.parquet"
    try:
        write_factor_long_table_qlib(factor_frames, engine, factor_long_path, args.force, logger)
    except FileExistsError as exc:
        logger.error("%s", exc)
        return 1

    outputs = {
        "features_base": str(qlib_dir / "features" / "base"),
        "labels": str(qlib_dir / "labels"),
        "meta": str(qlib_dir / "meta"),
        "factor_dir": str(qlib_dir / "features" / FACTOR_NAME_RET_LOG),
        "factor_long": str(factor_long_path),
    }
    snapshot_payload = build_snapshot_payload_qlib(args.run_id, sorted(assets.keys()), summary_rows, outputs, freq_token)
    snapshot_path = run_dir / "config_snapshots" / freq_token / args.run_id / "step1_qlib_summary.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Snapshot recorded at %s", snapshot_path)
    logger.info("Step1 Qlib conversion completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_qlib())
