#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib  # type: ignore

    matplotlib.use("Agg")  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib is an optional runtime dependency
    plt = None  # type: ignore

LOGGER = logging.getLogger(__name__)

_PLOTLY_AVAILABLE = False
_PLOTLY_IMPORT_ERROR: Optional[str] = None
_plotly_io = None
_model_performance_graph = None
_score_ic_graph = None

try:  # pragma: no cover - optional dependency
    import plotly.io as _plotly_io  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
    _PLOTLY_IMPORT_ERROR = str(exc)
else:  # pragma: no cover - optional dependency
    try:
        from qlib.contrib.report.analysis_model import (  # type: ignore
            model_performance_graph as _model_performance_graph,
        )
        from qlib.contrib.report.analysis_position import (  # type: ignore
            score_ic_graph as _score_ic_graph,
        )
    except Exception as exc:  # pragma: no cover - dependency missing
        _PLOTLY_IMPORT_ERROR = str(exc)
    else:
        _PLOTLY_AVAILABLE = True


@dataclass
class FigureRecord:
    path: str
    kind: str
    segment: str
    horizon: Optional[int]
    title: str
    format: str
    source: str
    fallback_used: bool
    html_path: Optional[str] = None


class ReportPathManager:
    """Utility to manage report output paths under a fixed root."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def ensure(self, relative: Path | str) -> Path:
        candidate = Path(relative)
        if not candidate.is_absolute():
            candidate = (self.root / candidate).resolve()
        if not candidate.is_relative_to(self.root):  # type: ignore[attr-defined]
            raise ValueError(f"Attempted to access path outside report root: {candidate}")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def relpath(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root))


def _load_segments(qlib_dir: Path, run_id: str) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    segments_path = qlib_dir / f"segments_{run_id}.json"
    segments: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    if not segments_path.exists():
        return segments
    try:
        payload = json.loads(segments_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to parse segments file %s: %s", segments_path, exc)
        return segments
    for name, window in payload.items():
        if isinstance(window, (list, tuple)) and len(window) == 2:
            start = pd.Timestamp(window[0])
            end = pd.Timestamp(window[1])
            if start.tzinfo is None:
                start = start.tz_localize("UTC")
            else:
                start = start.tz_convert("UTC")
            if end.tzinfo is None:
                end = end.tz_localize("UTC")
            else:
                end = end.tz_convert("UTC")
            if start <= end:
                segments[name] = (start, end)
    return segments


def _load_symbols(qlib_dir: Path, run_id: str, features_dir: Path) -> List[str]:
    symbols: List[str] = []
    txt_path = qlib_dir / f"instruments_{run_id}.txt"
    if txt_path.exists():
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            token = line.strip().split()
            if token:
                symbols.append(token[0])
    if not symbols:
        csv_path = qlib_dir / f"instruments_{run_id}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                LOGGER.warning("Failed to read instruments csv %s: %s", csv_path, exc)
            else:
                if "symbol" in df.columns:
                    symbols.extend(df["symbol"].dropna().astype(str).tolist())
    if not symbols:
        symbols = sorted({path.stem for path in features_dir.glob("*.parquet")})
    return sorted(set(symbols))


def _load_factor_table(features_dir: Path, symbol: str, factor_name: str) -> Optional[pd.DataFrame]:
    path = features_dir / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        LOGGER.warning("Failed to read factor parquet %s: %s", path, exc)
        return None
    if df.empty:
        return None
    column_options = [f"${factor_name}", factor_name]
    selected_col = next((col for col in column_options if col in df.columns), None)
    if selected_col is None:
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_cols:
            LOGGER.warning("No numeric factor column detected in %s", path)
            return None
        selected_col = numeric_cols[0]
    df = df.reset_index()
    if "instrument" in df.columns:
        df = df.rename(columns={"instrument": "symbol"})
    if "datetime" not in df.columns:
        LOGGER.warning("Datetime column missing in factor file %s", path)
        return None
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    df = df.rename(columns={selected_col: "score"})
    return df[["datetime", "symbol", "score"]]


def _load_label_table(labels_dir: Path, symbol: str) -> Optional[pd.DataFrame]:
    path = labels_dir / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        LOGGER.warning("Failed to read label parquet %s: %s", path, exc)
        return None
    if df.empty:
        return None
    df = df.reset_index()
    if "instrument" in df.columns:
        df = df.rename(columns={"instrument": "symbol"})
    if "datetime" not in df.columns:
        LOGGER.warning("Datetime column missing in label file %s", path)
        return None
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    return df


def _build_panel(
    features_dir: Path,
    labels_dir: Path,
    factor_name: str,
    horizons: Sequence[int],
    symbols: Sequence[str],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    horizon_columns = [f"label_ret_{int(h)}" for h in horizons]
    for symbol in symbols:
        factor_df = _load_factor_table(features_dir, symbol, factor_name)
        label_df = _load_label_table(labels_dir, symbol)
        if factor_df is None or label_df is None:
            continue
        merged = factor_df.merge(label_df, on=["datetime", "symbol"], how="left")
        for horizon in horizons:
            label_col = f"label_ret_{int(horizon)}"
            if label_col not in merged.columns:
                LOGGER.debug("Label column %s missing for %s", label_col, symbol)
                continue
            subset = (
                merged[["datetime", "symbol", "score", label_col]]
                .rename(columns={label_col: "label"})
                .dropna(subset=["score", "label"])
            )
            if subset.empty:
                continue
            subset["horizon"] = int(horizon)
            frames.append(subset)
    if not frames:
        raise ValueError("No matching factor/label data available for report generation.")
    panel = pd.concat(frames, ignore_index=True)
    panel["datetime"] = pd.to_datetime(panel["datetime"], utc=True)
    return panel


def _filter_panel(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (panel["datetime"] >= start) & (panel["datetime"] <= end)
    return panel.loc[mask].copy()


def _panel_to_pred_label(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    subset = panel.loc[panel["horizon"] == int(horizon), ["symbol", "datetime", "score", "label"]].copy()
    if subset.empty:
        return pd.DataFrame(columns=["score", "label"])
    subset = subset.dropna(subset=["score", "label"])
    if subset.empty:
        return pd.DataFrame(columns=["score", "label"])
    subset = subset.rename(columns={"symbol": "instrument"})
    subset = subset.sort_values(["instrument", "datetime"])
    pred_label = subset.set_index(["instrument", "datetime"])[["score", "label"]]
    index_values = []
    for inst, dt in pred_label.index.to_list():
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        index_values.append((inst, ts))
    pred_label.index = pd.MultiIndex.from_tuples(index_values, names=["instrument", "datetime"])
    return pred_label


def _daily_long_short(subset: pd.DataFrame, group_count: int = 5) -> pd.Series:
    results: Dict[pd.Timestamp, float] = {}
    if subset.empty:
        return pd.Series(results, dtype=float)
    for ts, day_df in subset.groupby("datetime"):
        day_sorted = day_df.sort_values("score", ascending=False)
        if len(day_sorted) < 2:
            results[pd.Timestamp(ts)] = np.nan
            continue
        bucket = max(len(day_sorted) // group_count, 1)
        top = day_sorted.head(bucket)["label"].mean()
        bottom = day_sorted.tail(bucket)["label"].mean()
        results[pd.Timestamp(ts)] = top - bottom
    return pd.Series(results).sort_index()


def _ic_series(pred_label: pd.DataFrame, method: str = "pearson") -> pd.Series:
    if pred_label.empty:
        return pd.Series(dtype=float)
    series = pred_label.groupby(level="datetime", group_keys=False).apply(
        lambda frame: frame["label"].corr(frame["score"], method=method)
    )
    return series.sort_index()


def _compute_metrics(subset: pd.DataFrame, pred_label: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "samples": int(len(pred_label)),
        "dates": int(subset["datetime"].nunique()),
        "symbols": int(subset["symbol"].nunique()),
    }
    if pred_label.empty:
        return metrics
    ic = _ic_series(pred_label, method="pearson")
    rank_ic = _ic_series(pred_label, method="spearman")
    ls = _daily_long_short(subset)
    metrics.update(
        {
            "ic_mean": float(ic.mean()) if not ic.empty else math.nan,
            "ic_std": float(ic.std(ddof=0)) if not ic.empty else math.nan,
            "ic_ir": float(ic.mean() / ic.std(ddof=0)) if (not ic.empty and ic.std(ddof=0) not in (0, np.nan)) else math.nan,
            "rank_ic_mean": float(rank_ic.mean()) if not rank_ic.empty else math.nan,
            "rank_ic_std": float(rank_ic.std(ddof=0)) if not rank_ic.empty else math.nan,
            "rank_ic_ir": float(rank_ic.mean() / rank_ic.std(ddof=0))
            if (not rank_ic.empty and rank_ic.std(ddof=0) not in (0, np.nan))
            else math.nan,
            "long_short_cum": float(ls.cumsum().iloc[-1]) if not ls.empty else math.nan,
            "long_short_avg": float(ls.mean()) if not ls.empty else math.nan,
        }
    )
    return metrics


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug or "figure"


def _canonical_kind(slug: str, source: str) -> str:
    mapping = {
        "cumulative_return": "long_short",
        "group_return": "long_short",
        "long_short": "long_short",
        "score_ic": "ic",
        "ic": "ic",
        "monthly_ic": "ic_monthly",
        "qq": "ic_qq",
        "distribution": "ic_distribution",
        "autocorr": "autocorrelation",
        "turnover": "turnover",
    }
    for key, value in mapping.items():
        if key in slug:
            return value
    if source == "analysis_position":
        return "ic"
    return slug or source


def _write_fallback_figure(kind: str, path: Path, pred_label: pd.DataFrame) -> None:
    if plt is None:  # pragma: no cover - matplotlib unavailable
        path.write_bytes(b"")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    if kind.startswith("ic"):
        data = _ic_series(pred_label)
        ax.set_ylabel("IC")
    elif kind.startswith("long_short"):
        subset = pred_label.reset_index().rename(columns={"instrument": "symbol"})
        subset["datetime"] = pd.to_datetime(subset["datetime"], utc=True)
        ls = _daily_long_short(subset)
        data = ls.cumsum()
        ax.set_ylabel("Long-Short")
    else:
        data = pred_label.groupby(level="datetime")["score"].mean()
        ax.set_ylabel("Score")
    if data is None or data.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.plot(data.index, data.values)
        ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"{kind.replace('_', ' ').title()} (fallback)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _export_plotly_figure(
    manager: ReportPathManager,
    base_name: str,
    figure,
    kind: str,
    pred_label: pd.DataFrame,
) -> Tuple[Path, Optional[Path], bool]:
    png_path = manager.ensure(f"{base_name}.png")
    html_path: Optional[Path] = None
    fallback_used = False
    if _PLOTLY_AVAILABLE and figure is not None:
        try:
            html_path = manager.ensure(f"{base_name}.html")
            _plotly_io.write_html(figure, str(html_path), include_plotlyjs="cdn", full_html=True)
        except Exception as exc:  # pragma: no cover - export failure
            LOGGER.warning("Failed to export plotly html for %s: %s", base_name, exc)
            html_path = None
        try:
            _plotly_io.write_image(figure, str(png_path), format="png", engine="kaleido")
        except Exception as exc:  # pragma: no cover - export failure
            LOGGER.warning("Falling back to matplotlib for %s: %s", base_name, exc)
            fallback_used = True
    else:
        fallback_used = True
    if fallback_used:
        _write_fallback_figure(kind, png_path, pred_label)
        if html_path is None:
            html_path = manager.ensure(f"{base_name}.html")
            html_path.write_text(
                f"<html><body><p>No interactive figure available for {base_name}.</p>"
                f"<img src='{png_path.name}' alt='{kind}'/></body></html>",
                encoding="utf-8",
            )
    return png_path, html_path, fallback_used


def _render_summary_html(
    manager: ReportPathManager,
    factor_name: str,
    run_id: str,
    horizons: Sequence[int],
    segments_payload: Dict[str, Dict[str, Any]],
    figures: List[FigureRecord],
    metrics: Dict[str, Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("<html><head><meta charset='utf-8'><title>Qlib Factor Analysis</title>")
    lines.append(
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px;}"
        "table{border-collapse:collapse;margin-bottom:24px;}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;}"
        "h2{margin-top:32px;}figure{margin:0 0 24px 0;}figcaption{font-size:0.9em;color:#555;}</style></head><body>"
    )
    lines.append(f"<h1>Factor Analysis Report &mdash; {factor_name} ({run_id})</h1>")
    lines.append("<section><h2>Overview</h2><ul>")
    lines.append(f"<li>Horizons: {', '.join(str(int(h)) for h in horizons)}</li>")
    lines.append(f"<li>Generated: {datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()}</li>")
    if not _PLOTLY_AVAILABLE:
        lines.append(
            f"<li>Interactive exports unavailable: {_PLOTLY_IMPORT_ERROR or 'plotly dependency missing'} (fallback figures generated)</li>"
        )
    lines.append("</ul></section>")
    lines.append("<section><h2>Segments</h2>")
    lines.append("<table><thead><tr><th>Segment</th><th>Start</th><th>End</th><th>Notes</th></tr></thead><tbody>")
    for name, payload in segments_payload.items():
        lines.append(
            f"<tr><td>{name}</td><td>{payload['start']}</td><td>{payload['end']}</td>"
            f"<td>{payload.get('note', '')}</td></tr>"
        )
    lines.append("</tbody></table></section>")
    lines.append("<section><h2>Key Metrics</h2>")
    for segment, per_horizon in metrics.items():
        if not per_horizon:
            continue
        lines.append(f"<h3>{segment}</h3>")
        lines.append(
            "<table><thead><tr><th>Horizon</th><th>Samples</th><th>Dates</th><th>Symbols</th>"
            "<th>IC Mean</th><th>Rank IC Mean</th><th>Long-Short Cum</th></tr></thead><tbody>"
        )
        for horizon, values in sorted(per_horizon.items(), key=lambda item: int(item[0])):
            lines.append(
                "<tr>"
                f"<td>{horizon}</td>"
                f"<td>{values.get('samples', '-')}</td>"
                f"<td>{values.get('dates', '-')}</td>"
                f"<td>{values.get('symbols', '-')}</td>"
                f"<td>{values.get('ic_mean', float('nan')):.4f}</td>"
                f"<td>{values.get('rank_ic_mean', float('nan')):.4f}</td>"
                f"<td>{values.get('long_short_cum', float('nan')):.4f}</td>"
                "</tr>"
            )
        lines.append("</tbody></table>")
    lines.append("</section>")
    if figures:
        lines.append("<section><h2>Figures</h2>")
        for record in figures:
            lines.append("<figure>")
            lines.append(
                f"<img src='{record.path}' alt='{record.kind}' style='max-width:960px;width:100%;height:auto;'/>"
            )
            caption = (
                f"Segment: {record.segment} | Horizon: {record.horizon if record.horizon is not None else 'n/a'} | "
                f"Kind: {record.kind}"
            )
            if record.fallback_used:
                caption += " | Rendered with fallback"
            lines.append(f"<figcaption>{caption}</figcaption>")
            lines.append("</figure>")
        lines.append("</section>")
    lines.append("</body></html>")
    return "".join(lines)


def _render_summary_markdown(
    manager: ReportPathManager,
    factor_name: str,
    run_id: str,
    horizons: Sequence[int],
    segments_payload: Dict[str, Dict[str, Any]],
    figures: List[FigureRecord],
    metrics: Dict[str, Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append(f"# Factor Analysis Report — {factor_name} ({run_id})\n")
    lines.append("## Overview\n")
    lines.append(f"- Horizons: {', '.join(str(int(h)) for h in horizons)}\n")
    timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    lines.append(f"- Generated: {timestamp}\n")
    if not _PLOTLY_AVAILABLE:
        lines.append(f"- Interactive exports unavailable: {_PLOTLY_IMPORT_ERROR or 'plotly dependency missing'}\n")
    lines.append("\n## Segments\n")
    lines.append("Segment | Start | End | Notes\n")
    lines.append("---|---|---|---\n")
    for name, payload in segments_payload.items():
        lines.append(
            f"{name} | {payload['start']} | {payload['end']} | {payload.get('note', '')}\n"
        )
    lines.append("\n## Key Metrics\n")
    for segment, per_horizon in metrics.items():
        if not per_horizon:
            continue
        lines.append(f"### {segment}\n")
        lines.append("Horizon | Samples | Dates | Symbols | IC Mean | Rank IC Mean | Long-Short Cum\n")
        lines.append("---|---|---|---|---|---|---\n")
        for horizon, values in sorted(per_horizon.items(), key=lambda item: int(item[0])):
            lines.append(
                f"{horizon} | {values.get('samples', '-')} | {values.get('dates', '-')} | {values.get('symbols', '-')} | "
                f"{values.get('ic_mean', float('nan')):.4f} | {values.get('rank_ic_mean', float('nan')):.4f} | "
                f"{values.get('long_short_cum', float('nan')):.4f}\n"
            )
    if figures:
        lines.append("\n## Figures\n")
        for record in figures:
            lines.append(f"![{record.kind}]({record.path})\n")
            caption = (
                f"Segment: {record.segment} | Horizon: {record.horizon if record.horizon is not None else 'n/a'} | "
                f"Kind: {record.kind}"
            )
            if record.fallback_used:
                caption += " | Rendered with fallback"
            lines.append(f"*{caption}*\n\n")
    return "".join(lines)


def generate_report(
    run_dir: Path | str,
    factor_name: str,
    horizons: Sequence[int],
    segments: Sequence[str] | None = None,
    output_dir: Path | str | None = None,
    output_format: str = "html",
) -> Dict[str, Any]:
    """Generate analytical plots and summary artefacts from Qlib outputs."""

    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_path}")
    run_id = run_path.name
    qlib_root = run_path / "qlib"
    if not qlib_root.exists():
        raise FileNotFoundError(f"qlib directory not found: {qlib_root}")
    candidate_dirs = [qlib_root]
    try:
        candidate_dirs.extend(sorted(path for path in qlib_root.iterdir() if path.is_dir()))
    except FileNotFoundError:
        pass
    selected_dir = None
    features_dir = None
    search_paths: List[Path] = []
    for candidate in candidate_dirs:
        candidate_features = candidate / "features" / factor_name
        search_paths.append(candidate_features)
        if candidate_features.exists():
            selected_dir = candidate
            features_dir = candidate_features
            break
    if selected_dir is None or features_dir is None:
        searched = ", ".join(str(path) for path in search_paths) or str(qlib_root / "features" / factor_name)
        raise FileNotFoundError(f"factor feature directory not found: {searched}")
    qlib_dir = selected_dir
    labels_dir = qlib_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")
    horizons_list = [int(h) for h in horizons]
    if not horizons_list:
        raise ValueError("At least one horizon is required for analysis.")

    symbols = _load_symbols(qlib_dir, run_id, features_dir)
    if not symbols:
        raise ValueError("No instruments detected for analysis.")

    panel = _build_panel(features_dir, labels_dir, factor_name, horizons_list, symbols)
    start_full = panel["datetime"].min()
    end_full = panel["datetime"].max()

    segment_windows = {"full": (start_full, end_full)}
    segment_windows.update(_load_segments(qlib_dir, run_id))

    if segments:
        selected_segments = []
        for name in segments:
            if name in segment_windows and name not in selected_segments:
                selected_segments.append(name)
    else:
        selected_segments = list(segment_windows.keys())

    if not selected_segments:
        raise ValueError("No valid segments selected for analysis.")

    if output_dir is None:
        output_base = run_path / "reports_qlib" / factor_name / "analysis"
    else:
        output_candidate = Path(output_dir)
        if output_candidate.is_absolute():
            output_base = output_candidate
        else:
            output_base = run_path / "reports_qlib" / factor_name / output_candidate
    manager = ReportPathManager(output_base)

    figure_records: List[FigureRecord] = []
    summary_metrics: Dict[str, Dict[str, Any]] = {}
    per_segment_payload: Dict[str, Dict[str, Any]] = {}
    counters: Dict[str, int] = {}

    for segment_name in selected_segments:
        window = segment_windows.get(segment_name)
        if window is None:
            continue
        segment_panel = _filter_panel(panel, window[0], window[1])
        per_segment_payload[segment_name] = {
            "start": window[0].isoformat(),
            "end": window[1].isoformat(),
            "note": "",
        }
        summary_metrics[segment_name] = {}
        if segment_panel.empty:
            LOGGER.warning("Segment %s has no data between %s and %s", segment_name, window[0], window[1])
            continue
        for horizon in horizons_list:
            pred_label = _panel_to_pred_label(segment_panel, horizon)
            subset = segment_panel.loc[segment_panel["horizon"] == int(horizon)]
            if pred_label.empty or subset.empty:
                LOGGER.debug("No data for segment %s horizon %s", segment_name, horizon)
                continue
            metrics = _compute_metrics(subset, pred_label)
            summary_metrics[segment_name][str(horizon)] = metrics
            figures_candidates: List[Tuple[str, str, Any]] = []
            if _PLOTLY_AVAILABLE and _model_performance_graph is not None:
                try:
                    mp_figures = _model_performance_graph(
                        pred_label,
                        show_notebook=False,
                        show_nature_day=False,
                    )
                except Exception as exc:  # pragma: no cover - runtime failure
                    LOGGER.warning("model_performance_graph failed for %s/%s: %s", segment_name, horizon, exc)
                    mp_figures = []
                for fig in mp_figures:
                    title = getattr(fig.layout.title, "text", None) or str(fig.layout.title or "")
                    figures_candidates.append(("analysis_model", title or "model_performance", fig))
            if _PLOTLY_AVAILABLE and _score_ic_graph is not None:
                try:
                    position_figures = _score_ic_graph(pred_label, show_notebook=False)
                except Exception as exc:  # pragma: no cover - runtime failure
                    LOGGER.debug("score_ic_graph failed for %s/%s: %s", segment_name, horizon, exc)
                    position_figures = []
                for fig in position_figures:
                    title = getattr(fig.layout.title, "text", None) or str(fig.layout.title or "")
                    figures_candidates.append(("analysis_position", title or "score_ic", fig))
            if not figures_candidates:
                figures_candidates.append(("fallback", "ic", None))
            seen_titles: set[str] = set()
            for source, title, figure in figures_candidates:
                slug = _slugify(title)
                if slug in seen_titles:
                    continue
                seen_titles.add(slug)
                kind = _canonical_kind(slug, source)
                base_name = f"analysis_{kind}_{segment_name}"
                if len(horizons_list) > 1:
                    base_name += f"_h{int(horizon)}"
                counters.setdefault(base_name, 0)
                counter = counters[base_name]
                counters[base_name] += 1
                if counter:
                    base_with_index = f"{base_name}_{counter}"
                else:
                    base_with_index = base_name
                png_path, html_path, fallback_used = _export_plotly_figure(
                    manager,
                    base_with_index,
                    figure,
                    kind,
                    pred_label,
                )
                record = FigureRecord(
                    path=manager.relpath(png_path),
                    kind=kind,
                    segment=segment_name,
                    horizon=int(horizon),
                    title=title or kind,
                    format="png",
                    source=source,
                    fallback_used=fallback_used,
                    html_path=manager.relpath(html_path) if html_path else None,
                )
                figure_records.append(record)

    summary_data = {
        "run_id": run_id,
        "factor_name": factor_name,
        "horizons": horizons_list,
        "segments": per_segment_payload,
        "figures": [asdict(record) for record in figure_records],
        "metrics": summary_metrics,
        "generated_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "plotly_available": _PLOTLY_AVAILABLE,
        "plotly_error": _PLOTLY_IMPORT_ERROR,
    }

    summary_path = manager.ensure("analysis_summary.json")
    summary_path.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8")

    if output_format.lower() == "html":
        content = _render_summary_html(manager, factor_name, run_id, horizons_list, per_segment_payload, figure_records, summary_metrics)
        summary_doc = manager.ensure(f"analysis_{factor_name}_{run_id}.html")
    elif output_format.lower() == "md":
        content = _render_summary_markdown(manager, factor_name, run_id, horizons_list, per_segment_payload, figure_records, summary_metrics)
        summary_doc = manager.ensure(f"analysis_{factor_name}_{run_id}.md")
    else:
        raise ValueError("output_format must be either 'html' or 'md'")
    summary_doc.write_text(content, encoding="utf-8")

    result = {
        "run_id": run_id,
        "factor_name": factor_name,
        "output_dir": str(manager.root),
        "summary_path": str(summary_doc),
        "summary_json": str(summary_path),
        "figures": [asdict(record) for record in figure_records],
        "metrics": summary_metrics,
        "segments": per_segment_payload,
        "plotly_available": _PLOTLY_AVAILABLE,
        "plotly_error": _PLOTLY_IMPORT_ERROR,
    }
    return result

