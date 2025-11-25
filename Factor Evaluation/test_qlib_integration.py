import json
import subprocess
import sys
from pathlib import Path
from qlib_report_analysis import generate_report

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def _run(command):
    subprocess.run([sys.executable] + command, check=True, cwd=ROOT)


def test_qlib_pipeline(tmp_path):
    output_root = tmp_path / "AI_Agent_Output_fix"

    for freq, pandas_freq in [("1min", "T"), ("3min", "3T")]:
        run_id = f"test_run_{freq.replace('min', 'm')}"
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True)
        freq_dir = run_dir / freq
        freq_dir.mkdir(parents=True, exist_ok=True)

        manifest = [{"symbol": "BTCUSDT", "date": "2024-01-01", "status": "completed"}]
        summary = {
            "run_id": run_id,
            "mode": "execute",
            "start": "2024-01-01T00:00:00+00:00",
            "end": "2024-01-02T00:00:00+00:00",
            "symbols": ["BTCUSDT"],
            "dates": ["2024-01-01"],
            "total_tasks": 1,
            "counts": {"completed": 1},
            "audit_ready_flag": True,
        }
        summary["bar_freq"] = freq
        (freq_dir / "step0_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (freq_dir / "step0_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        _run(["step0_prepare_inputs_qlib_01.py", "--output-dir", str(output_root), "--run-id", run_id, "--bar-freq", freq])

        golden_dir = run_dir / "golden" / freq
        golden_dir.mkdir(parents=True, exist_ok=True)
        timestamps = pd.date_range("2024-01-01", periods=8, freq=pandas_freq, tz="UTC")
        base_df = pd.DataFrame(
            {
                "open": np.linspace(10, 17, len(timestamps)),
                "high": np.linspace(11, 18, len(timestamps)),
                "low": np.linspace(9, 16, len(timestamps)),
                "close": np.linspace(10.5, 17.5, len(timestamps)),
                "volume": np.linspace(1000, 2000, len(timestamps)),
            },
            index=timestamps,
        )
        base_df.index.name = "open_time"
        base_df.to_parquet(golden_dir / "base_BTCUSDT_20240101.parquet")

        ratios_5 = np.linspace(1.01, 1.02, len(timestamps))
        ratios_15 = np.linspace(1.02, 1.03, len(timestamps))
        labels_df = pd.DataFrame(
            {
                "label_ret_5": np.log(ratios_5),
                "label_ret_15": np.log(ratios_15),
            },
            index=timestamps,
        )
        labels_df.index.name = "open_time"
        labels_df.to_parquet(golden_dir / "labels_BTCUSDT_20240101.parquet")

        factor_df = pd.DataFrame({"RET_LOG_1": np.linspace(0.0, 0.01, len(timestamps))}, index=timestamps)
        factor_df.index.name = "open_time"
        factor_df.to_parquet(golden_dir / "factor_RET_LOG_1_BTCUSDT_20240101.parquet")

        _run(["step1_contract_and_golden_qlib_02.py", "--output-dir", str(output_root), "--run-id", run_id, "--bar-freq", freq])

        universe_dir = run_dir / "universe" / freq
        universe_dir.mkdir(parents=True, exist_ok=True)
        universe_df = pd.DataFrame(
            [
                {
                    "symbol": "BTCUSDT",
                    "coverage": 1.0,
                    "excluded": False,
                    "start_date_utc": "2024-01-01T00:00:00+00:00",
                    "end_date_utc": "2024-01-02T00:00:00+00:00",
                    "first_time_utc": "2024-01-01T00:00:00+00:00",
                    "last_time_utc": "2024-01-01T00:21:00+00:00" if freq == "3min" else "2024-01-01T00:07:00+00:00",
                    "expected_rows": len(timestamps),
                    "actual_rows": len(timestamps),
                }
            ]
        )
        universe_df.to_csv(universe_dir / f"universe_{run_id}.csv", index=False)

        _run(["step2_loader_and_qc_qlib_03.py", "--output-dir", str(output_root), "--run-id", run_id, "--bar-freq", freq])

        factor_name = "demo"
        factor_dir = run_dir / "factors" / freq / factor_name
        factor_dir.mkdir(parents=True, exist_ok=True)
        factor_series = pd.Series(np.linspace(0.1, 0.8, len(timestamps)), index=timestamps)
        factor_series.index.name = "open_time"
        factor_series.to_frame(name=factor_name).to_parquet(factor_dir / "BTCUSDT.parquet")

        qlib_factor_dir = run_dir / "qlib" / freq / "features" / factor_name
        qlib_factor_dir.mkdir(parents=True, exist_ok=True)
        multi_index = pd.MultiIndex.from_arrays(
            [timestamps, ["BTCUSDT"] * len(timestamps)],
            names=["datetime", "instrument"],
        )
        qlib_factor_df = pd.DataFrame({f"${factor_name}": factor_series.values}, index=multi_index)
        qlib_factor_df.to_parquet(qlib_factor_dir / "BTCUSDT.parquet")
        long_df = qlib_factor_df.reset_index().rename(columns={f"${factor_name}": factor_name})
        (run_dir / "qlib" / freq / "features").mkdir(parents=True, exist_ok=True)
        long_df.to_parquet(run_dir / "qlib" / freq / "features" / f"{factor_name}_long.parquet", index=False)

        _run(
            [
                "step4_factor_evaluation_qlib_05.py",
                "--output-dir",
                str(output_root),
                "--factor-name",
                factor_name,
                "--run-id",
                run_id,
                "--bar-freq",
                freq,
                "--horizons",
                "5,15",
                "--comparison-threshold",
                "0.5",
            ]
        )

        # Verify logs reference the correct frequency without leaking other values
        log_path = run_dir / "logs" / freq / f"step4_qlib_{run_id}.log"
        log_text = log_path.read_text(encoding="utf-8")
        assert f"bar_freq={freq}" in log_text
        if freq == "3min":
            assert "bar_freq=1min" not in log_text
        qlib_root = run_dir / "qlib" / freq
        assert (qlib_root / f"calendar_{run_id}.csv").exists()
        assert (qlib_root / "features" / "base" / "BTCUSDT.parquet").exists()
        reports_root = run_dir / "reports_qlib" / freq / factor_name / run_id
        comparison_path = reports_root / "comparison.json"
        assert comparison_path.exists()
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        assert "threshold" in comparison
        summary_path = run_dir / "config_snapshots" / freq / run_id / "step4_qlib_summary.json"
        assert summary_path.exists()

        analysis_result = generate_report(
            run_dir=run_dir,
            factor_name=factor_name,
            horizons=[5, 15],
            segments=["full"],
            output_dir=None,
            output_format="html",
        )
        assert Path(analysis_result["summary_path"]).exists()
        assert Path(analysis_result["summary_json"]).exists()
        assert analysis_result["figures"], "Expected at least one analysis figure"
        first_fig = Path(analysis_result["output_dir"]) / analysis_result["figures"][0]["path"]
        assert first_fig.exists()
