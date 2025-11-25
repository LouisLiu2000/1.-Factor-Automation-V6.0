"""Utility script to inspect a finished backtest run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a completed backtest run.")
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier, e.g. topk_60min_62_2c528388bb_mfi14_topk_test.",
    )
    parser.add_argument(
        "--results-root",
        default=Path("Backtest/results"),
        type=Path,
        help="Root directory storing backtest outputs. Defaults to Backtest/results.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate Matplotlib charts (daily equity, daily returns histogram, drawdown). Requires matplotlib.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory to save plots (defaults to the run's output directory).",
    )
    return parser.parse_args()


def load_index(results_root: Path) -> list[dict]:
    index_path = results_root / "index.json"
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    if not isinstance(data, list):
        return []
    return data


def locate_run_directory(run_id: str, results_root: Path) -> tuple[Path, Optional[str]]:
    entries = load_index(results_root)
    for entry in entries:
        if entry.get("run_id") == run_id:
            path = Path(entry.get("results_path", "")).expanduser()
            if path.exists():
                return path, entry.get("strategy")
    # Fallback: search recursively
    for strategy_dir in results_root.iterdir():
        if not strategy_dir.is_dir():
            continue
        candidate = strategy_dir / run_id
        if candidate.exists():
            return candidate, strategy_dir.name
    raise FileNotFoundError(f"Could not locate results directory for run_id='{run_id}'.")


def compute_daily_equity(equity_df: pd.DataFrame) -> pd.DataFrame:
    series = equity_df.set_index("datetime")["equity"]
    daily_equity = series.resample("D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    cumulative = (daily_returns + 1.0).cumprod()
    rolling_max = daily_equity.cummax()
    drawdown = (daily_equity / rolling_max) - 1.0
    result = pd.DataFrame(
        {
            "equity": daily_equity,
            "daily_return": daily_returns,
            "cumulative": cumulative,
            "drawdown": drawdown,
        }
    )
    return result


def main() -> int:
    args = parse_args()
    results_dir, strategy = locate_run_directory(args.run_id, args.results_root.resolve())
    metrics_path = results_dir / "metrics.json"
    context_path = results_dir / "run_context.json"
    equity_path = results_dir / "equity_curve.csv"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    equity_df = pd.read_csv(equity_path, parse_dates=["datetime"])

    daily_stats = compute_daily_equity(equity_df)
    max_drawdown = daily_stats["drawdown"].min()
    avg_daily_return = daily_stats["daily_return"].mean()
    std_daily_return = daily_stats["daily_return"].std()
    daily_summary = daily_stats.copy()
    daily_summary["daily_return_pct"] = daily_summary["daily_return"] * 100
    daily_summary["drawdown_pct"] = daily_summary["drawdown"] * 100
    summary_path = results_dir / f"{args.run_id}_daily_summary.csv"
    daily_summary.to_csv(summary_path, index_label="date")

    print("=" * 80)
    print(f"Run ID       : {args.run_id}")
    print(f"Strategy     : {strategy or context.get('config', {}).get('strategy')}")
    print(f"Results Dir  : {results_dir}")
    print(f"Factor Set   : {context.get('bundle', {}).get('factor_set')}")
    print(f"Timeframe    : {context.get('bundle', {}).get('timeframe')}")
    print(f"Symbols      : {', '.join(context.get('bundle', {}).get('symbols', []))}")
    print("-" * 80)
    print("Headline Metrics (from metrics.json)")
    for key, value in metrics.items():
        print(f"  {key:20}: {value}")
    print("-" * 80)
    print("Daily Equity Summary")
    print(f"  Start Date       : {daily_stats.index.min().date()}")
    print(f"  End Date         : {daily_stats.index.max().date()}")
    print(f"  Max Drawdown (%) : {max_drawdown * 100:.2f}")
    print(f"  Avg Daily Ret (%) : {avg_daily_return * 100:.3f}")
    print(f"  Std Daily Ret (%) : {std_daily_return * 100:.3f}")
    print(f"  Daily summary CSV : {summary_path}")
    print("-" * 80)
    print("Recent Daily Equity")
    tail = daily_stats.tail(10).copy()
    tail["equity"] = tail["equity"].round(2)
    tail["daily_return"] = (tail["daily_return"] * 100).round(3)
    tail["drawdown"] = (tail["drawdown"] * 100).round(2)
    print(tail[["equity", "daily_return", "drawdown"]])
    print("-" * 80)
    print("Top 10 Daily Drawdowns (%)")
    drawdown_tail = daily_stats.sort_values("drawdown").head(10).copy()
    drawdown_tail["drawdown"] = (drawdown_tail["drawdown"] * 100).round(2)
    print(drawdown_tail[["drawdown"]])
    print("-" * 80)
    print("Daily Return Distribution (percentage bins)")
    hist = pd.cut(
        daily_stats["daily_return"],
        bins=[-0.5, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1, 0.5],
    ).value_counts().sort_index()
    for interval, count in hist.items():
        print(f"  {interval}: {count}")
    if args.plot:
        plot_dir = args.plot_dir or results_dir
        plot_dir.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            print("matplotlib is required for plotting but is not installed:", exc)
        else:
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
            daily_stats["equity"].plot(ax=axes[0], title="Daily Equity")
            axes[0].set_ylabel("Equity")
            daily_stats["daily_return"].mul(100).plot(
                ax=axes[1],
                title="Daily Returns (%)",
            )
            axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
            axes[1].set_ylabel("Return (%)")
            daily_stats["drawdown"].mul(100).plot(
                ax=axes[2],
                title="Daily Drawdown (%)",
                color="tab:red",
            )
            axes[2].set_ylabel("Drawdown (%)")
            axes[2].set_xlabel("Date")
            plt.tight_layout()
            output_path = plot_dir / f"{args.run_id}_analysis.png"
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            print("=" * 80)
            print(f"Saved charts to: {output_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
