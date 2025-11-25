#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import math

from frequency_utils import (
    BarFrequency,
    add_bar_frequency_argument,
    parse_bar_frequency_or_exit,
)

try:
    from qlib.contrib.eva import alpha as eva_alpha
    from qlib.contrib import evaluate as qlib_evaluate
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "pyqlib is required for Qlib evaluation. Install it via `pip install pyqlib`."
    ) from exc

from step4_factor_evaluation import (
    DEFAULT_GROUPS,
    DEFAULT_TIME_SPLIT,
    DEFAULT_TRIM_Q,
    SafePathManager,
    compute_group_returns,
    compute_t_p,
    evaluation_local,
)

from qlib_report_analysis import generate_report

DEFAULT_THRESHOLD = 0.05


def parse_args_qlib() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate factor performance using Qlib artifacts.")
    parser.add_argument("--output-dir", required=True, help="Root directory containing run outputs.")
    parser.add_argument("--factor-name", required=True, help="Name of the factor to evaluate.")
    parser.add_argument("--run-id", required=True, help="Run identifier matching previous steps.")
    parser.add_argument("--horizons", default="5,15,35", help="Comma separated horizons (default: 5,15,35).")
    parser.add_argument("--group", type=int, default=DEFAULT_GROUPS, help="Quantile group count for analysis.")
    parser.add_argument(
        "--time-split",
        choices=["month", "quarter"],
        default=DEFAULT_TIME_SPLIT,
        help="Temporal split used for stability analysis (default: month).",
    )
    parser.add_argument(
        "--segments",
        help="Optional override path to segments JSON. Defaults to qlib/segments_<run_id>.json.",
    )
    parser.add_argument(
        "--comparison-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Relative threshold for flagging differences between local and Qlib metrics (default: 0.05).",
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Skip running the local evaluation for comparison.",
    )
    parser.add_argument(
        "--enable-report-analysis",
        action="store_true",
        help="Generate extended analysis report and visualizations.",
    )
    parser.add_argument(
        "--analysis-output-dir",
        default="analysis",
        help="Sub-directory (relative to reports_qlib/factor) for analysis artifacts (default: analysis).",
    )
    parser.add_argument(
        "--analysis-format",
        choices=["html", "md"],
        default="html",
        help="Format for the consolidated analysis summary (default: html).",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    add_bar_frequency_argument(parser)
    return parser.parse_args()


def setup_logger_qlib(
    manager: SafePathManager, run_id: str, level: str, bar_frequency: BarFrequency
) -> logging.Logger:
    log_path = manager.prepare_path(Path("logs") / bar_frequency.canonical / f"step4_qlib_{run_id}.log")
    logger = logging.getLogger(f"step4_qlib_{run_id}")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    manager.set_logger(logger)
    return logger


def parse_horizons_qlib(raw: str) -> List[int]:
    horizons = [int(token.strip()) for token in raw.split(",") if token.strip()]
    return horizons or [5, 15, 35]


def load_segments_qlib(
    qlib_dir: Path,
    run_id: str,
    override: Optional[str],
    logger: logging.Logger,
) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    if override:
        segments_path = Path(override).expanduser().resolve()
    else:
        segments_path = qlib_dir / f"segments_{run_id}.json"
    segments: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    if segments_path.exists():
        payload = json.loads(segments_path.read_text(encoding="utf-8"))
        for name, values in payload.items():
            if isinstance(values, (list, tuple)) and len(values) == 2:
                start = pd.Timestamp(values[0]).tz_localize("UTC") if pd.Timestamp(values[0]).tzinfo is None else pd.Timestamp(values[0]).tz_convert("UTC")
                end = pd.Timestamp(values[1]).tz_localize("UTC") if pd.Timestamp(values[1]).tzinfo is None else pd.Timestamp(values[1]).tz_convert("UTC")
                segments[name] = (start, end)
    else:
        logger.warning("Segments file not found at %s; proceeding without predefined splits.", segments_path)
    return segments


def read_instrument_list_qlib(qlib_dir: Path, run_id: str) -> List[str]:
    txt_path = qlib_dir / f"instruments_{run_id}.txt"
    symbols: List[str] = []
    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as handle:
            for line in handle:
                tokens = line.strip().split()
                if tokens:
                    symbols.append(tokens[0])
    else:
        csv_path = qlib_dir / f"instruments_{run_id}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "symbol" in df.columns:
                symbols = df["symbol"].dropna().astype(str).tolist()
    return sorted(set(symbols))


def load_factor_data_qlib(
    factor_dir: Path,
    symbol: str,
    factor_name: str,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    path = factor_dir / f"{symbol}.parquet"
    if not path.exists():
        logger.warning("Factor file missing for %s: %s", symbol, path)
        return None
    df = pd.read_parquet(path)
    if df.empty:
        logger.warning("Factor file empty for %s", symbol)
        return None
    column_name = f"${factor_name}"
    if column_name not in df.columns:
        raise SystemExit(f"Factor column {column_name} not found in {path}")
    df = df.rename(columns={column_name: "factor"}).reset_index()
    if "instrument" in df.columns:
        df = df.rename(columns={"instrument": "symbol"})
    if "datetime" not in df.columns:
        raise SystemExit(f"Datetime column missing in {path}")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    return df[["datetime", "symbol", "factor"]]


def load_label_data_qlib(
    labels_dir: Path,
    symbol: str,
    horizons: Sequence[int],
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    path = labels_dir / f"{symbol}.parquet"
    if not path.exists():
        logger.warning("Label file missing for %s: %s", symbol, path)
        return None
    df = pd.read_parquet(path)
    if df.empty:
        logger.warning("Label file empty for %s", symbol)
        return None
    df = df.reset_index()
    if "instrument" in df.columns:
        df = df.rename(columns={"instrument": "symbol"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    required_cols = [f"label_ret_{h}" for h in horizons]
    available = [col for col in required_cols if col in df.columns]
    missing = sorted(set(required_cols) - set(available))
    if missing:
        logger.warning("Label columns %s missing for %s", missing, symbol)
    if not available:
        return None
    return df[["datetime", "symbol"] + available]


def build_panel_qlib(
    factor_dir: Path,
    labels_dir: Path,
    symbols: Sequence[str],
    factor_name: str,
    horizons: Sequence[int],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[str]]:
    panel_frames: List[pd.DataFrame] = []
    symbol_summaries: List[Dict[str, Any]] = []
    processed: List[str] = []
    for symbol in symbols:
        factor_df = load_factor_data_qlib(factor_dir, symbol, factor_name, logger)
        if factor_df is None:
            continue
        label_df = load_label_data_qlib(labels_dir, symbol, horizons, logger)
        if label_df is None:
            continue
        merged = factor_df.merge(label_df, on=["datetime", "symbol"], how="left")
        for horizon in horizons:
            label_col = f"label_ret_{horizon}"
            if label_col not in merged.columns:
                logger.warning("Label %s missing for %s after merge", label_col, symbol)
                continue
            records = pd.DataFrame(
                {
                    "datetime": merged["datetime"],
                    "symbol": merged["symbol"],
                    "horizon": horizon,
                    "factor": merged["factor"],
                    "label": merged[label_col],
                }
            )
            records["mask"] = (
                records["factor"].notna()
                & records["label"].notna()
                & np.isfinite(records["factor"])
                & np.isfinite(records["label"])
            )
            panel_frames.append(records)
            symbol_summaries.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "rows": int(len(records)),
                    "valid_samples": int(records["mask"].sum()),
                    "valid_rate": float(records["mask"].mean() if len(records) else 0.0),
                }
            )
        processed.append(symbol)
    if not panel_frames:
        raise SystemExit("No usable factor/label pairs were found for Qlib evaluation.")
    panel = pd.concat(panel_frames, ignore_index=True)
    panel["datetime"] = pd.to_datetime(panel["datetime"], utc=True)
    return panel, symbol_summaries, processed


def filter_panel_by_range_qlib(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    mask = (panel["datetime"] >= start) & (panel["datetime"] <= end)
    return panel[mask].copy()


def compute_stability_qlib(
    subset: pd.DataFrame,
    horizon: int,
    time_split: str,
) -> List[Dict[str, Any]]:
    if subset.empty:
        return []
    if time_split == "month":
        periods = subset["datetime"].dt.to_period("M")
    else:
        periods = subset["datetime"].dt.to_period("Q")
    subset = subset.assign(period=periods)
    records: List[Dict[str, Any]] = []
    for period, frame in subset.groupby("period"):
        valid = frame[frame["mask"]]
        if valid["symbol"].nunique() < 2 or len(valid) < 2:
            continue
        rank_ic = valid["factor"].corr(valid["label"], method="spearman")
        if pd.isna(rank_ic):
            continue
        records.append(
            {
                "horizon": horizon,
                "period": str(period),
                "rank_ic": float(rank_ic),
                "samples": int(valid["mask"].sum()),
            }
        )
    return records


def evaluate_segment_qlib(
    panel: pd.DataFrame,
    horizons: Sequence[int],
    group_count: int,
    time_split: str,
) -> Dict[str, Any]:
    def _to_serializable_number(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (np.floating, float, int)):
            value = float(value)
        else:
            if pd.isna(value):
                return None
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
        if math.isnan(value):
            return None
        return value

    metric_rows: List[Dict[str, Any]] = []
    for horizon in horizons:
        subset = panel[(panel["horizon"] == horizon) & panel["mask"]].copy()
        subset.sort_values("datetime", inplace=True)
        metrics: Dict[str, Any] = {"horizon": horizon, "samples": int(len(subset))}
        if subset.empty:
            metrics.update(
                {
                    "ic": float("nan"),
                    "rank_ic": float("nan"),
                    "ic_t": None,
                    "ic_p": None,
                    "ic_mean": float("nan"),
                    "ic_std": 0.0,
                    "ir": None,
                    "rank_ic_mean": float("nan"),
                    "rank_ic_std": 0.0,
                    "rank_ir": None,
                    "stability": [],
                    "groups": [],
                    "long_short": [],
                    "risk": {},
                }
            )
            metric_rows.append(metrics)
            continue
        factor_series = subset.set_index(["datetime", "symbol"])["factor"].copy()
        factor_series.index = factor_series.index.set_names(["datetime", "instrument"])
        label_series = subset.set_index(["datetime", "symbol"])["label"].copy()
        label_series.index = label_series.index.set_names(["datetime", "instrument"])
        ic_series, rank_series = eva_alpha.calc_ic(factor_series, label_series, dropna=True)
        metrics["ic"] = float(subset["factor"].corr(subset["label"], method="pearson"))
        metrics["rank_ic"] = float(subset["factor"].corr(subset["label"], method="spearman"))
        metrics["ic_mean"] = float(ic_series.mean()) if not ic_series.empty else float("nan")
        metrics["ic_std"] = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
        metrics["rank_ic_mean"] = float(rank_series.mean()) if not rank_series.empty else float("nan")
        metrics["rank_ic_std"] = float(rank_series.std(ddof=1)) if len(rank_series) > 1 else 0.0
        metrics["ir"] = (
            metrics["ic_mean"] / metrics["ic_std"] if metrics["ic_std"] not in (0.0, None) else None
        )
        metrics["rank_ir"] = (
            metrics["rank_ic_mean"] / metrics["rank_ic_std"]
            if metrics["rank_ic_std"] not in (0.0, None)
            else None
        )
        t_value, p_value = compute_t_p(metrics["ic"], int(subset["mask"].sum()))
        metrics["ic_t"] = t_value
        metrics["ic_p"] = p_value
        metrics["stability"] = compute_stability_qlib(subset, horizon, time_split)
        group_records, ls_records = compute_group_returns(subset, horizon, group_count)
        metrics["groups"] = group_records
        metrics["long_short"] = ls_records
        if ls_records:
            ls_df = pd.DataFrame(ls_records).sort_values("datetime")
            ls_series = pd.Series(
                ls_df["long_short"].values,
                index=pd.to_datetime(ls_df["datetime"], utc=True),
            )
            risk_stats = qlib_evaluate.risk_analysis(ls_series, freq="day")
            if not risk_stats.empty:
                risk_series = risk_stats.squeeze()
                if isinstance(risk_series, pd.Series):
                    iterator = risk_series.items()
                elif isinstance(risk_series, (list, tuple, np.ndarray)):
                    iterator = enumerate(risk_series)
                else:
                    iterator = [("value", risk_series)]
                risk_dict: Dict[str, Optional[float]] = {}
                for key, value in iterator:
                    risk_dict[str(key)] = _to_serializable_number(value)
            else:
                risk_dict = {}
            metrics["risk"] = risk_dict
        else:
            metrics["risk"] = {}
        metric_rows.append(metrics)
    ic_summary = pd.DataFrame(
        [
            {
                "horizon": m["horizon"],
                "samples": m["samples"],
                "ic": m.get("ic"),
                "rank_ic": m.get("rank_ic"),
                "ic_t": m.get("ic_t"),
                "ic_p": m.get("ic_p"),
                "ic_mean": m.get("ic_mean"),
                "ic_std": m.get("ic_std"),
                "ir": m.get("ir"),
                "rank_ic_mean": m.get("rank_ic_mean"),
                "rank_ic_std": m.get("rank_ic_std"),
                "rank_ir": m.get("rank_ir"),
            }
            for m in metric_rows
        ]
    )
    group_df = pd.DataFrame([rec for m in metric_rows for rec in m.get("groups", [])])
    if not group_df.empty:
        group_summary = (
            group_df.groupby(["horizon", "group"], as_index=False)["mean_return"].mean().rename(
                columns={"mean_return": "avg_return"}
            )
        )
    else:
        group_summary = pd.DataFrame(columns=["horizon", "group", "avg_return"])
    stability_df = pd.DataFrame([rec for m in metric_rows for rec in m.get("stability", [])])
    long_short_df = pd.DataFrame([rec for m in metric_rows for rec in m.get("long_short", [])])
    risk_summary = {m["horizon"]: m.get("risk", {}) for m in metric_rows if m.get("risk")}
    return {
        "metrics": metric_rows,
        "ic_summary": ic_summary,
        "group_summary": group_summary,
        "stability": stability_df,
        "long_short": long_short_df,
        "risk": risk_summary,
    }


def write_segment_outputs_qlib(
    manager: SafePathManager,
    report_root: Path,
    factor_name: str,
    segment: str,
    result: Dict[str, Any],
) -> None:
    report_dir = report_root / segment
    ic_path = manager.prepare_path(report_dir / f"ic_summary_{segment}.csv")
    result["ic_summary"].to_csv(ic_path, index=False)
    group_path = manager.prepare_path(report_dir / f"group_return_summary_{segment}.csv")
    result["group_summary"].to_csv(group_path, index=False)
    stability_df = result.get("stability", pd.DataFrame())
    if not stability_df.empty:
        stability_path = manager.prepare_path(report_dir / f"rankic_stability_{segment}.csv")
        stability_df.to_csv(stability_path, index=False)
    long_short_df = result.get("long_short", pd.DataFrame())
    if not long_short_df.empty:
        long_short_df = long_short_df.sort_values(["horizon", "datetime"])
        ls_path = manager.prepare_path(report_dir / f"long_short_curve_{segment}.csv")
        long_short_df.to_csv(ls_path, index=False)
    risk_summary = result.get("risk", {})
    risk_path = manager.prepare_path(report_dir / f"risk_summary_{segment}.json")
    risk_path.write_text(json.dumps(risk_summary, ensure_ascii=False, indent=2), encoding="utf-8")


def compare_with_local_qlib(
    local_result: Dict[str, Any],
    qlib_ic_summary: pd.DataFrame,
    threshold: float,
) -> List[Dict[str, Any]]:
    comparisons: List[Dict[str, Any]] = []
    local_df = local_result.get("ic_summary")
    if local_df is None or local_df.empty or qlib_ic_summary.empty:
        return comparisons
    for _, q_row in qlib_ic_summary.iterrows():
        horizon = int(q_row["horizon"])
        local_row = local_df[local_df["horizon"] == horizon]
        if local_row.empty:
            continue
        local_row = local_row.iloc[0]
        for column in ["ic_mean", "rank_ic_mean", "ir", "rank_ir"]:
            q_value = q_row.get(column)
            l_value = local_row.get(column)
            if pd.isna(q_value) or pd.isna(l_value):
                continue
            abs_diff = float(q_value - l_value)
            denom = max(abs(float(l_value)), 1e-12)
            rel_diff = abs(abs_diff) / denom
            exceeds = rel_diff > threshold
            comparisons.append(
                {
                    "horizon": horizon,
                    "metric": column,
                    "qlib": float(q_value),
                    "local": float(l_value),
                    "absolute_diff": abs_diff,
                    "relative_diff": rel_diff,
                    "exceeds_threshold": exceeds,
                }
            )
    return comparisons


def write_summary_snapshot_qlib(
    manager: SafePathManager, snapshot_dir: Path, payload: Dict[str, Any]
) -> None:
    snapshot_path = manager.prepare_path(snapshot_dir / "step4_qlib_summary.json")
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main_qlib() -> int:
    args = parse_args_qlib()
    bar_frequency = parse_bar_frequency_or_exit(args.bar_freq)
    setattr(args, "bar_freq_canonical", bar_frequency.canonical)
    freq_token = bar_frequency.canonical
    run_dir = Path(args.output_dir).expanduser().resolve() / args.run_id
    preferred_qlib_dir = run_dir / "qlib" / freq_token
    legacy_qlib_dir = run_dir / "qlib"
    if preferred_qlib_dir.exists():
        qlib_dir = preferred_qlib_dir
        qlib_legacy_fallback = False
    elif legacy_qlib_dir.exists():
        qlib_dir = legacy_qlib_dir
        qlib_legacy_fallback = True
    else:
        raise SystemExit(f"Qlib directory not found for bar_freq={freq_token}: tried {preferred_qlib_dir} and {legacy_qlib_dir}")

    manager = SafePathManager(run_dir)
    logger = setup_logger_qlib(manager, args.run_id, args.log_level, bar_frequency)
    if qlib_legacy_fallback:
        logger.warning(
            "Qlib directory for bar_freq=%s missing at %s; falling back to legacy path %s",
            freq_token,
            preferred_qlib_dir,
            legacy_qlib_dir,
        )
    logger.info("Evaluating factor %s in run_id=%s bar_freq=%s", args.factor_name, args.run_id, freq_token)
    logger.info("Labels use log returns: log(close_t+h / close_t).")
    horizons = parse_horizons_qlib(args.horizons)
    group_count = max(2, int(args.group))

    features_dir = qlib_dir / "features"
    factor_dir = features_dir / args.factor_name
    labels_dir = qlib_dir / "labels"
    if not factor_dir.exists():
        raise SystemExit(f"Factor features directory not found: {factor_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    reports_root = Path("reports_qlib") / freq_token / args.factor_name / args.run_id
    snapshot_dir = Path("config_snapshots") / freq_token / args.run_id

    symbols = read_instrument_list_qlib(qlib_dir, args.run_id)
    if not symbols:
        symbols = sorted(path.stem for path in factor_dir.glob("*.parquet"))
    if not symbols:
        raise SystemExit("No symbols detected for evaluation.")

    logger.info("Building panel for %d symbols", len(symbols))
    panel, symbol_summaries, processed_symbols = build_panel_qlib(
        factor_dir,
        labels_dir,
        symbols,
        args.factor_name,
        horizons,
        logger,
    )

    if panel.empty:
        raise SystemExit("Panel is empty after aggregation; aborting.")

    segments = load_segments_qlib(qlib_dir, args.run_id, args.segments, logger)
    full_start = panel["datetime"].min()
    full_end = panel["datetime"].max()
    segments_with_full: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {"full": (full_start, full_end)}
    segments_with_full.update(segments)

    evaluation_results: Dict[str, Dict[str, Any]] = {}
    for name, (start, end) in segments_with_full.items():
        logger.info("Evaluating segment %s: %s -> %s", name, start, end)
        segment_panel = filter_panel_by_range_qlib(panel, start, end)
        if segment_panel.empty:
            logger.warning("Segment %s has no data; skipping.", name)
            continue
        result = evaluate_segment_qlib(segment_panel, horizons, group_count, args.time_split)
        evaluation_results[name] = result
        write_segment_outputs_qlib(manager, reports_root, args.factor_name, name, result)

    comparison_records: List[Dict[str, Any]] = []
    local_result: Optional[Dict[str, Any]] = None
    if not args.skip_local:
        logger.info("Running local evaluation for comparison (results will not be re-written).")
        local_args = argparse.Namespace(
            output_dir=str(run_dir),
            factor_name=args.factor_name,
            horizons=",".join(str(h) for h in horizons),
            group=group_count,
            trim_extreme="false",
            trim_quantile=DEFAULT_TRIM_Q,
            time_split=args.time_split,
            run_id=args.run_id,
            bar_freq=freq_token,
            bar_freq_canonical=freq_token,
        )
        local_result = evaluation_local(local_args, write_outputs=False)
        full_result = evaluation_results.get("full")
        if local_result and full_result:
            comparison_records = compare_with_local_qlib(
                local_result,
                full_result["ic_summary"],
                float(args.comparison_threshold),
            )
            for record in comparison_records:
                if record.get("exceeds_threshold"):
                    logger.warning(
                        "Metric %s horizon %s differs beyond threshold: rel diff=%.4f",
                        record["metric"],
                        record["horizon"],
                        record["relative_diff"],
                    )

    comparison_path = manager.prepare_path(reports_root / "comparison.json")
    comparison_payload = {
        "threshold": float(args.comparison_threshold),
        "records": comparison_records,
    }
    comparison_path.write_text(json.dumps(comparison_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    analysis_result = None
    if args.enable_report_analysis:
        logger.info("Generating extended analysis report (format=%s, output=%s)", args.analysis_format, args.analysis_output_dir)
        try:
            analysis_result = generate_report(
                run_dir=run_dir,
                factor_name=args.factor_name,
                horizons=horizons,
                segments=list(segments_with_full.keys()),
                output_dir=args.analysis_output_dir,
                output_format=args.analysis_format,
            )
        except Exception as exc:
            logger.exception("Failed to generate extended analysis report: %s", exc)
        else:
            logger.info("Extended analysis report saved to %s", analysis_result.get("summary_path"))

    summary_payload = {
        "run_id": args.run_id,
        "factor": args.factor_name,
        "bar_freq": freq_token,
        "horizons": list(horizons),
        "group_count": group_count,
        "time_split": args.time_split,
        "segments": {
            name: {"start": start.isoformat(), "end": end.isoformat()}
            for name, (start, end) in segments_with_full.items()
        },
        "symbols": processed_symbols,
        "evaluation_results": {
            name: {
                "ic_summary": result["ic_summary"].to_dict(orient="records"),
                "group_summary": result["group_summary"].to_dict(orient="records"),
                "stability": result["stability"].to_dict(orient="records"),
                "risk": result["risk"],
            }
            for name, result in evaluation_results.items()
        },
        "comparison": comparison_records,
        "reports_dir": str(reports_root),
        "local_evaluation_ran": bool(local_result is not None),
    }
    if analysis_result:
        summary_payload["analysis_report"] = {
            "summary": analysis_result.get("summary_path"),
            "summary_json": analysis_result.get("summary_json"),
            "output_dir": analysis_result.get("output_dir"),
            "figures": analysis_result.get("figures", []),
            "plotly_available": analysis_result.get("plotly_available"),
            "plotly_error": analysis_result.get("plotly_error"),
        }

    write_summary_snapshot_qlib(manager, snapshot_dir, summary_payload)
    logger.info("Qlib evaluation completed for factor %s run_id=%s", args.factor_name, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_qlib())
