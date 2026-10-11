"""Run a fixed, shared REMUS baseline sensitivity matrix without overwriting results.

Command delay is a configured simulation-time proxy: execution holds the
previous reference until issue_time + delay. It does not reproduce measured
wall-clock computation or asynchronous real-time middleware.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("empty3d_baseline", "empty3d", {}),
    ("crossing3d_baseline", "crossing3d", {}),
    ("crossing3d_command_delay_0p5", "crossing3d", {"command-delay-s": .5}),
    ("crossing3d_command_delay_1p0", "crossing3d", {"command-delay-s": 1.0}),
    ("crossing3d_fov90x6", "crossing3d", {"sensor-hfov-deg": 90., "sensor-vfov-deg": 6.}),
    ("crossing3d_control_dt1", "crossing3d", {"control-dt": 1.}),
    ("multi_crossing3d_fov90x6", "multi_crossing3d", {"sensor-hfov-deg": 90., "sensor-vfov-deg": 6.}),
)
METRICS = (
    "backend", "scenario", "seed", "status", "reached_goal", "failure_reason",
    "duration_s", "goal_distance_m", "path_length_m", "minimum_model_clearance_m",
    "safety_validation_status", "safety_failure_reason", "actual_safety_margin_met",
    "actual_close_interaction_count", "maximum_simultaneous_close_interactions",
    "observed_unique_count", "maximum_observed_obstacles", "maximum_tracked_obstacles",
    "mean_observed_obstacles", "maximum_used_observation_age_s", "emergency_count",
    "unsafe_emergency_forecast_count", "planning_ms_median", "planning_ms_p95",
    "planning_ms_max", "planning_exceeds_control_period_count",
    "prediction_position_rmse_m", "prediction_position_max_error_m",
    "all_states_finite", "wall_clock_s", "physics_dt_s", "control_dt_s",
    "candidate_budget", "maximum_full_rollouts_per_plan", "command_delay_s",
)


def run_study(out, selected=None, max_time=150., seed=7):
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; choose a new study directory")
    selected = set(selected or (name for name, _, _ in CASES))
    unknown = selected - {name for name, _, _ in CASES}
    if unknown:
        raise ValueError(f"unknown sensitivity cases: {sorted(unknown)}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    commands = []
    start = time.perf_counter()
    for name, scenario, differences in CASES:
        if name not in selected:
            continue
        case_out = out / name
        args = [sys.executable, str(ROOT / "run_stage2.py"), "simulate", "--backend", "remus6dof",
                "--scenario", scenario, "--config", str(ROOT / "configs/stage2_development.json"),
                "--out", str(case_out), "--seed", str(seed), "--horizon-s", "24",
                "--candidate-budget", "9", "--max-time", str(max_time),
                "--command-delay-s", "0", "--control-dt", "2",
                "--sensor-hfov-deg", "120", "--sensor-vfov-deg", "30", "--no-animation"]
        # Explicit overrides replace the existing value, keeping one unambiguous
        # occurrence per option in the actual command log.
        for option, value in differences.items():
            args[args.index("--" + option) + 1] = str(value)
        # One full graphic set suffices; preserve all actual state/control CSVs
        # for every case without spending experiment time repeatedly plotting.
        if name != "crossing3d_baseline":
            args.append("--no-plot")
        print(f"START {name}", flush=True)
        started = time.perf_counter()
        log = out / f"{name}_command.log"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       check=False, text=True, encoding="utf-8",
                                       env={**os.environ, "PYTHONUTF8": "1"})
        record = {"case": name, "arguments": args, "working_directory": str(ROOT),
                  "exit_code": completed.returncode, "wall_clock_s": time.perf_counter() - started,
                  "console_log": log.name, "differences_from_common_baseline": differences}
        commands.append(record)
        summary_path = case_out / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        config = json.loads((case_out / "config.json").read_text(encoding="utf-8")) if (case_out / "config.json").exists() else {}
        row = {"case": name, "process_exit_code": completed.returncode,
               "sensor_hfov_deg": config.get("sensor_hfov_deg"),
               "sensor_vfov_deg": config.get("sensor_vfov_deg"),
               "configured_command_delay_s": config.get("command_delay_s"),
               **{key: summary.get(key) for key in METRICS}}
        rows.append(row)
        with (out / "sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        (out / "commands.json").write_text(json.dumps(commands, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"DONE {name}: exit={completed.returncode}, status={summary.get('status', 'program_error')}, "
              f"clearance={summary.get('minimum_model_clearance_m')}, "
              f"P95_ms={summary.get('planning_ms_p95')}", flush=True)
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "study": "single_seed_REMUS_baseline_timing_and_FOV_sensitivity",
        "backend": "remus6dof", "common_seed": seed, "common_horizon_s": 24.,
        "common_candidate_budget": 9, "common_max_time_s": max_time,
        "common_physics_dt_s": .02, "default_control_dt_s": 2.,
        "default_sensor_fov_deg": [120., 30.],
        "command_delay_interpretation": "fixed simulation-time proxy holding previous reference until plan issue time + configured delay; not measured wall-clock real-time execution",
        "case_count": len(rows), "program_failures": sum(row["process_exit_code"] != 0 for row in rows),
        "task_successes": sum(row.get("reached_goal") is True for row in rows),
        "finite_horizon_model_check_passes": sum(row.get("safety_validation_status") == "passed_finite_horizon_model_checks" for row in rows),
        "planning_period_budget_exceeded_cases": sum((row.get("planning_exceeds_control_period_count") or 0) > 0 for row in rows),
        "task_failures_are_preserved": True,
        "comparison_scope": "same scene geometry, seed, original REMUS dynamics and collision standard; one disclosed parameter changed per crossing sensitivity case",
        "wall_clock_s": time.perf_counter() - start, "rows": rows,
    }
    (out / "study_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results_stage2/review_20261011_sensitivity")
    parser.add_argument("--cases", nargs="+", choices=[name for name, _, _ in CASES])
    parser.add_argument("--max-time", type=float, default=150.)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        result = run_study(args.out, args.cases, args.max_time, args.seed)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["program_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
