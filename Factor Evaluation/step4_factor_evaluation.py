#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from frequency_utils import (
    BarFrequency,
    add_bar_frequency_argument,
    parse_bar_frequency_or_exit,
)

try:  # optional dependency
    from scipy import stats as scipy_stats  # type: ignore
except ImportError:  # pragma: no cover
    scipy_stats = None

DEFAULT_HORIZONS = [5, 15, 35]
DEFAULT_GROUPS = 5
DEFAULT_TRIM_Q = 0.01
DEFAULT_TIME_SPLIT = "month"



class SecurityViolation(RuntimeError):
    pass


class SafePathManager:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root.resolve()
        self.logger: Optional[logging.Logger] = None

    def set_logger(self, logger: logging.Logger) -> None:
        self.logger = logger

    def prepare_path(self, relative_path: Path | str) -> Path:
        target = self._resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def resolve(self, relative_path: Path | str) -> Path:
        return self._resolve(relative_path)

    def _resolve(self, relative_path: Path | str) -> Path:
        candidate = Path(relative_path)
        resolved = (self.output_root / candidate).resolve()
        if not resolved.is_relative_to(self.output_root):
            self._log_violation(resolved)
            raise SecurityViolation(f"Attempted I/O outside OUTPUT_DIR: {resolved}")
        return resolved

    def _log_violation(self, target: Path) -> None:
        message = f"SECURITY_VIOLATION: attempted write to {target}"
        if self.logger:
            self.logger.error(message)
        else:
            print(message)


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


def safe_write_parquet(manager: SafePathManager, df: pd.DataFrame, relative_path: Path | str) -> Path:
    target = manager.prepare_path(relative_path)
    engine = detect_parquet_engine()
    df.to_parquet(target, engine=engine)
    return target


def safe_write_csv(manager: SafePathManager, df: pd.DataFrame, relative_path: Path | str) -> Path:
    target = manager.prepare_path(relative_path)
    df.to_csv(target, index=False)
    return target


def safe_write_json(manager: SafePathManager, payload: Dict[str, Any], relative_path: Path | str) -> Path:
    target = manager.prepare_path(relative_path)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return target


def safe_write_figure(manager: SafePathManager, fig: plt.Figure, relative_path: Path | str) -> Path:
    target = manager.prepare_path(relative_path)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def safe_write_html(manager: SafePathManager, html: str, relative_path: Path | str) -> Path:
    target = manager.prepare_path(relative_path)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(html)
    return target


def gather_environment_snapshot(extra_packages: Iterable[str]) -> Dict[str, Any]:
    import platform
    from importlib import metadata as importlib_metadata

    packages: Dict[str, Optional[str]] = {}
    for name in extra_packages:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Factor evaluation module")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--factor-name", required=True)
    parser.add_argument("--horizons", default="5,15,35")
    parser.add_argument("--group", type=int, default=DEFAULT_GROUPS)
    parser.add_argument("--trim-extreme", default="false")
    parser.add_argument("--trim-quantile", type=float, default=DEFAULT_TRIM_Q)
    parser.add_argument("--time-split", choices=["month", "quarter"], default=DEFAULT_TIME_SPLIT)
    parser.add_argument("--run-id")
    add_bar_frequency_argument(parser)
    return parser.parse_args()


def load_factor_series(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("Factor file empty")
    series = df.iloc[:, 0].copy()
    series.index = pd.to_datetime(series.index, utc=True)
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="first")]
    return series.astype(float)


def find_golden_base_files(
    output_root: Path, symbol: str, bar_frequency: BarFrequency, logger: logging.Logger
) -> List[Path]:
    preferred_dir = (output_root / "golden" / bar_frequency.canonical).resolve()
    if preferred_dir.exists():
        golden_dir = preferred_dir
    else:
        legacy_dir = (output_root / "golden").resolve()
        if legacy_dir.exists():
            logger.warning("Golden directory for bar_freq=%s not found at %s; falling back to legacy path %s", bar_frequency.canonical, preferred_dir, legacy_dir)
            golden_dir = legacy_dir
        else:
            return []
    pattern = f"base_{symbol}_*.parquet"
    return sorted(p for p in golden_dir.glob(pattern) if p.is_file())


def load_close_history(
    output_root: Path,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bar_frequency: BarFrequency,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    files = find_golden_base_files(output_root, symbol, bar_frequency, logger)
    if not files:
        logger.warning("No golden base files for %s", symbol)
        return None
    frames: List[pd.DataFrame] = []
    for path in files:
        df = pd.read_parquet(path)
        if "close" not in df.columns:
            continue
        sub = df.copy()
        sub.index = pd.to_datetime(sub.index, utc=True, errors="coerce")
        sub = sub[~sub.index.duplicated(keep="last")]
        frames.append(sub[["close"]])
    if not frames:
        logger.warning("Golden files for %s missing close column", symbol)
        return None
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    if combined.index.min() > start:
        logger.warning("Close history for %s starts after requested window", symbol)
        return None
    if end <= start:
        logger.warning("Requested window end %s not after start %s for %s", end, start, symbol)
        return None
    # Reconstruct the exact set of bars requested using the parsed offset.
    expected_index = pd.date_range(
        start=start,
        end=end - bar_frequency.delta,
        freq=bar_frequency.offset,
        tz="UTC",
    )
    expected_index.name = combined.index.name or "open_time"
    combined = combined.reindex(expected_index)
    combined["close"] = pd.to_numeric(combined["close"], errors="coerce")
    return combined


def generate_labels(close_df: pd.DataFrame, horizons: Sequence[int], bar_frequency: BarFrequency) -> pd.DataFrame:
    """Compute forward log returns for each horizon using the configured bar spacing."""
    labels = pd.DataFrame(index=close_df.index)
    labels.index.name = close_df.index.name
    close = pd.to_numeric(close_df["close"], errors="coerce")
    base_values = close.to_numpy()
    for horizon in horizons:
        delta = bar_frequency.to_timedelta(horizon)
        future_index = close.index + delta
        future_close = close.reindex(future_index)
        future_values = future_close.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = future_values / base_values
            log_returns = np.log(ratio)
        invalid = (~np.isfinite(log_returns)) | (base_values <= 0) | (future_values <= 0)
        log_returns[invalid] = np.nan
        labels[f"label_ret_{horizon}"] = pd.Series(log_returns, index=labels.index)
    return labels

def apply_trim(series: pd.Series, enabled: bool, quantile: float) -> pd.Series:
    if not enabled:
        return series
    lower, upper = series.quantile([quantile, 1 - quantile])
    return series.clip(lower=lower, upper=upper)


def compute_t_p(r: float, n: int) -> Tuple[Optional[float], Optional[float]]:
    if n < 3 or r is None or np.isnan(r) or abs(r) >= 1:
        return None, None
    t_value = r * math.sqrt((n - 2) / (1 - r * r))
    if scipy_stats is None:
        return t_value, None
    p_value = float(2 * scipy_stats.t.sf(abs(t_value), df=n - 2))
    return t_value, p_value


def per_time_correlations(df: pd.DataFrame, method: str) -> List[float]:
    values: List[float] = []
    for _, group in df.groupby("datetime"):
        group = group[group["mask"]]
        if group["symbol"].nunique() < 2 or len(group) < 2:
            continue
        corr = group[["factor", "label"]].corr(method=method).iloc[0, 1]
        if not np.isnan(corr):
            values.append(float(corr))
    return values


def compute_group_returns(subset: pd.DataFrame, horizon: int, group_count: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if subset.empty:
        return [], []
    group_records: List[Dict[str, Any]] = []
    ls_records: List[Dict[str, Any]] = []
    for dt, frame in subset.groupby("datetime"):
        frame = frame[frame["mask"]].copy()
        if len(frame) < group_count:
            continue
        try:
            frame["bucket"] = pd.qcut(frame["factor"], q=group_count, labels=False, duplicates="drop")
        except Exception:
            continue
        if frame["bucket"].nunique() < group_count:
            continue
        group_mean = frame.groupby("bucket")["label"].mean()
        for bucket, value in group_mean.items():
            group_records.append(
                {
                    "horizon": horizon,
                    "datetime": dt,
                    "group": int(bucket) + 1,
                    "mean_return": float(value),
                }
            )
        long_short = float(group_mean.max() - group_mean.min())
        ls_records.append({"horizon": horizon, "datetime": dt, "long_short": long_short})
    return group_records, ls_records


def compute_metrics(panel: pd.DataFrame, horizon: int, time_split: str, group_count: int) -> Dict[str, Any]:
    subset = panel[(panel["horizon"] == horizon) & panel["mask"]].copy()
    subset.sort_values("datetime", inplace=True)
    metrics: Dict[str, Any] = {"horizon": horizon, "samples": int(len(subset))}
    if subset.empty:
        metrics["groups"] = []
        metrics["long_short"] = []
        metrics["stability"] = []
        return metrics
    metrics["ic"] = float(subset["factor"].corr(subset["label"], method="pearson"))
    metrics["rank_ic"] = float(subset["factor"].corr(subset["label"], method="spearman"))
    t_value, p_value = compute_t_p(metrics["ic"], metrics["samples"])
    metrics["ic_t"] = t_value
    metrics["ic_p"] = p_value
    ic_series = per_time_correlations(subset, "pearson")
    rank_series = per_time_correlations(subset, "spearman")
    if ic_series:
        metrics["ic_mean"] = float(np.mean(ic_series))
        metrics["ic_std"] = float(np.std(ic_series, ddof=1)) if len(ic_series) > 1 else 0.0
        if metrics.get("ic_std") not in (0.0, None):
            metrics["ir"] = metrics["ic_mean"] / metrics["ic_std"]
    if rank_series:
        metrics["rank_ic_mean"] = float(np.mean(rank_series))
        metrics["rank_ic_std"] = float(np.std(rank_series, ddof=1)) if len(rank_series) > 1 else 0.0
        if metrics.get("rank_ic_std") not in (0.0, None):
            metrics["rank_ir"] = metrics["rank_ic_mean"] / metrics["rank_ic_std"]
    if time_split == "month":
        periods = subset["datetime"].dt.to_period("M")
    else:
        periods = subset["datetime"].dt.to_period("Q")
    subset = subset.assign(period=periods)
    stability_records: List[Dict[str, Any]] = []
    for period, frame in subset.groupby("period"):
        if frame["mask"].sum() < 2:
            continue
        rank_ic = frame["factor"].corr(frame["label"], method="spearman")
        if np.isnan(rank_ic):
            continue
        stability_records.append(
            {
                "horizon": horizon,
                "period": str(period),
                "rank_ic": float(rank_ic),
                "samples": int(frame["mask"].sum()),
            }
        )
    metrics["stability"] = stability_records
    group_records, ls_records = compute_group_returns(subset, horizon, group_count)
    metrics["groups"] = group_records
    metrics["long_short"] = ls_records
    return metrics


def build_ic_plot(summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    horizons = summary["horizon"].astype(str)
    ax.bar(horizons, summary["ic"], label="IC", alpha=0.7)
    ax.plot(horizons, summary["rank_ic"], marker="o", label="RankIC", color="orange")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Correlation")
    ax.legend()
    ax.set_title("IC vs RankIC by Horizon")
    return fig


def build_long_short_plot(ls_curve: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    for horizon, frame in ls_curve.groupby("horizon"):
        sorted_frame = frame.sort_values("datetime")
        cumulative = sorted_frame["long_short"].cumsum()
        ax.plot(sorted_frame["datetime"], cumulative, label=f"H{horizon}")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Cumulative LS Return")
    ax.legend()
    ax.set_title("Long-Short Cumulative Returns")
    return fig


def render_report(factor_name: str, run_id: str, summary: pd.DataFrame, group_summary: pd.DataFrame, stability: pd.DataFrame, plots: Dict[str, str]) -> str:
    stability_html = stability.to_html(index=False) if not stability.empty else "<p>No stability data</p>"
    return (
        "<html><head><meta charset='utf-8'><title>Factor Report</title></head><body>"
        + f"<h1>Factor Evaluation Report - {factor_name}</h1>"
        + f"<p>Run ID: {run_id}</p>"
        + "<h2>IC Summary</h2>" + summary.to_html(index=False)
        + "<h2>Group Return Summary</h2>" + group_summary.to_html(index=False)
        + "<h2>RankIC Stability</h2>" + stability_html
        + "<h2>Figures</h2>"
        + f"<div><img src='{plots['ic_plot']}' alt='IC Plot' width='600'></div>"
        + f"<div><img src='{plots['ls_plot']}' alt='Long Short Plot' width='600'></div>"
        + "</body></html>"
    )


def evaluation_local_qlib(args: argparse.Namespace, write_outputs: bool = True) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    if not horizons:
        horizons = DEFAULT_HORIZONS
    trim_extreme = parse_bool(args.trim_extreme)
    quantile = float(args.trim_quantile)
    run_id = args.run_id or pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")

    manager = SafePathManager(output_dir)
    bar_frequency = parse_bar_frequency_or_exit(getattr(args, "bar_freq_canonical", getattr(args, "bar_freq", "1min")))
    freq_token = bar_frequency.canonical
    log_path = manager.prepare_path(Path("logs") / freq_token / f"{run_id}.log")
    logger = logging.getLogger("step4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    manager.set_logger(logger)

    logger.info("Factor evaluation start: factor=%s run_id=%s bar_freq=%s", args.factor_name, run_id, bar_frequency.canonical)
    logger.info("Labels use log returns: log(close_t+h / close_t).")

    factor_dir = manager.resolve(Path("factors") / freq_token / args.factor_name)
    if not factor_dir.exists():
        raise SystemExit(f"Factor directory not found: {factor_dir}")
    factor_files = sorted(f for f in factor_dir.glob("*.parquet") if f.is_file())
    if not factor_files:
        raise SystemExit(f"No factor files present in {factor_dir}")

    labels_dir = Path("labels") / freq_token
    artifacts_dir = Path("artifacts") / freq_token
    report_dir_base = Path("reports") / freq_token / args.factor_name / run_id

    panel_rows: List[pd.DataFrame] = []
    symbol_summaries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    max_horizon = max(horizons)

    for factor_path in factor_files:
        symbol = factor_path.stem
        try:
            factor_series = load_factor_series(factor_path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", symbol, exc)
            skipped.append({"symbol": symbol, "reason": str(exc)})
            continue
        trimmed_factor = apply_trim(factor_series, trim_extreme, quantile)
        start_time = factor_series.index.min()
        end_time = factor_series.index.max()

        labels_path = manager.resolve(labels_dir / f"{symbol}_labels.parquet")
        labels_df: Optional[pd.DataFrame] = None
        recompute = True
        if labels_path.exists():
            try:
                labels_df = pd.read_parquet(labels_path)
                labels_df.index = pd.to_datetime(labels_df.index, utc=True)
                label_cols = [f"label_ret_{h}" for h in horizons]
                if all(col in labels_df.columns for col in label_cols):
                    if labels_df.index.min() <= start_time and labels_df.index.max() >= end_time:
                        recompute = False
            except Exception:
                recompute = True
        if recompute:
            history = load_close_history(
                manager.output_root,
                symbol,
                start_time,
                end_time + bar_frequency.to_timedelta(max_horizon),
                bar_frequency,
                logger,
            )
            if history is None:
                skipped.append({"symbol": symbol, "reason": "close history unavailable"})
                continue
            labels_df = generate_labels(history, horizons, bar_frequency)
            if write_outputs:
                safe_write_parquet(manager, labels_df, labels_dir / f"{symbol}_labels.parquet")
        assert labels_df is not None
        aligned_labels = labels_df.reindex(factor_series.index)
        for horizon in horizons:
            label_col = f"label_ret_{horizon}"
            if label_col not in aligned_labels.columns:
                logger.warning("Label %s missing for %s", label_col, symbol)
                continue
            records = pd.DataFrame(
                {
                    "datetime": factor_series.index,
                    "symbol": symbol,
                    "horizon": horizon,
                    "factor": trimmed_factor,
                    "label": aligned_labels[label_col],
                }
            )
            records["mask"] = (
                records["factor"].notna()
                & records["label"].notna()
                & np.isfinite(records["factor"])
                & np.isfinite(records["label"])
            )
            panel_rows.append(records)
            symbol_summaries.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "rows": int(len(records)),
                    "valid_samples": int(records["mask"].sum()),
                    "valid_rate": float(records["mask"].mean()),
                }
            )
        logger.info("Processed %s, range %s -> %s", symbol, start_time, end_time)

    if not panel_rows:
        raise SystemExit("No aligned data available for evaluation")

    panel = pd.concat(panel_rows, ignore_index=True)
    panel["datetime"] = pd.to_datetime(panel["datetime"], utc=True)
    logger.info("Panel rows: %d", len(panel))

    metrics_per_horizon: List[Dict[str, Any]] = []
    group_records: List[Dict[str, Any]] = []
    stability_records: List[Dict[str, Any]] = []
    ls_records: List[Dict[str, Any]] = []

    group_count = max(2, int(args.group))

    for horizon in horizons:
        metrics = compute_metrics(panel, horizon, args.time_split, group_count)
        metrics_per_horizon.append(metrics)
        group_records.extend(metrics.get("groups", []))
        stability_records.extend(metrics.get("stability", []))
        ls_records.extend(metrics.get("long_short", []))

    ic_summary = pd.DataFrame(
        [
            {
                "horizon": m.get("horizon"),
                "samples": m.get("samples"),
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
            for m in metrics_per_horizon
        ]
    )

    group_df = pd.DataFrame(group_records)
    if not group_df.empty:
        group_summary = (
            group_df.groupby(["horizon", "group"], as_index=False)
            ["mean_return"]
            .mean()
            .rename(columns={"mean_return": "avg_return"})
        )
    else:
        group_summary = pd.DataFrame(columns=["horizon", "group", "avg_return"])

    stability_df = pd.DataFrame(stability_records)
    ls_df = pd.DataFrame(ls_records)

    report_dir = report_dir_base
    if write_outputs:
        safe_write_csv(manager, ic_summary, report_dir / "ic_summary.csv")
        safe_write_csv(manager, group_summary, report_dir / "group_return_summary.csv")
        if not stability_df.empty:
            safe_write_csv(manager, stability_df, report_dir / "rankic_stability.csv")

        plots: Dict[str, str] = {}
        if not ic_summary.empty:
            ic_plot = build_ic_plot(ic_summary.fillna(0))
            plots["ic_plot"] = "ic_by_horizon.png"
            safe_write_figure(manager, ic_plot, report_dir / plots["ic_plot"])
        else:
            plots["ic_plot"] = "ic_by_horizon.png"
            empty_fig, ax = plt.subplots(figsize=(6, 3))
            ax.set_title("IC Plot (No Data)")
            ax.axis("off")
            safe_write_figure(manager, empty_fig, report_dir / plots["ic_plot"])
        if not ls_df.empty:
            ls_plot = build_long_short_plot(ls_df)
            plots["ls_plot"] = "long_short_curve.png"
            safe_write_figure(manager, ls_plot, report_dir / plots["ls_plot"])
        else:
            plots["ls_plot"] = "long_short_curve.png"
            empty_fig, ax = plt.subplots(figsize=(6, 3))
            ax.set_title("Long-Short Cumulative Returns (No Data)")
            ax.axis("off")
            safe_write_figure(manager, empty_fig, report_dir / plots["ls_plot"])

        html_report = render_report(
            args.factor_name,
            run_id,
            ic_summary.fillna(""),
            group_summary.fillna(""),
            stability_df.fillna("") if not stability_df.empty else pd.DataFrame(columns=["horizon", "period", "rank_ic", "samples"]),
            plots,
        )
        report_name = f"report_{args.factor_name}_{run_id}.html"
        safe_write_html(manager, html_report, report_dir / report_name)


    symbol_summary_df = pd.DataFrame(symbol_summaries)
    if write_outputs and not symbol_summary_df.empty:
        safe_write_csv(manager, symbol_summary_df, report_dir / "symbol_valid_summary.csv")

    if write_outputs and skipped:
        safe_write_csv(manager, pd.DataFrame(skipped), artifacts_dir / f"step4_skipped_{run_id}.csv")

    snapshot_dir = Path("config_snapshots") / freq_token / run_id
    env_snapshot = gather_environment_snapshot(["pandas", "numpy", "matplotlib", "scipy"])
    config_payload = {
        "run_id": run_id,
        "parameters": {
            "output_dir": str(output_dir),
            "factor_name": args.factor_name,
            "horizons": horizons,
            "group": group_count,
            "trim_extreme": trim_extreme,
            "trim_quantile": quantile,
            "time_split": args.time_split,
            "bar_freq": bar_frequency.canonical,
        },
        "metrics": ic_summary.to_dict(orient="records"),
        "group_summary": group_summary.to_dict(orient="records"),
        "stability": stability_df.to_dict(orient="records") if not stability_df.empty else [],
        "symbol_summary": symbol_summary_df.to_dict(orient="records"),
        "skipped": skipped,
        "environment": env_snapshot,
    }
    if write_outputs:
        safe_write_json(manager, config_payload, snapshot_dir / "step4_args.json")
    logger.info("Evaluation completed: factor=%s run_id=%s bar_freq=%s", args.factor_name, run_id, bar_frequency.canonical)

    result_payload = {
        "run_id": run_id,
        "factor_name": args.factor_name,
        "ic_summary": ic_summary,
        "group_summary": group_summary,
        "stability": stability_df,
        "long_short": ls_df,
        "symbol_summary": symbol_summary_df,
        "skipped": skipped,
        "metrics_per_horizon": metrics_per_horizon,
        "config": config_payload,
        "bar_freq": bar_frequency.canonical,
    }
    return result_payload


def evaluation_local(args: argparse.Namespace, write_outputs: bool = True) -> Dict[str, Any]:
    return evaluation_local_qlib(args, write_outputs)


def main() -> None:
    args = parse_arguments()
    bar_frequency = parse_bar_frequency_or_exit(args.bar_freq)
    setattr(args, "bar_freq_canonical", bar_frequency.canonical)
    evaluation_local_qlib(args, write_outputs=True)


if __name__ == "__main__":
    main()
