"""Preserve Stage3 raw evidence and reports in a fresh standalone review ZIP.

Ordinary ZIP compression only: no file/model/checksum identity mechanisms.
Failed navigation and failed test records are kept alongside successful runs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auv3d.stage2_config import Stage2Config
from scripts_stage3.run_experiments import ExperimentCase, metric_row


EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", "pytest_tmp", "pytest_temp", "pytest_work", "venv", "site-packages"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _within(path, parent):
    return path == parent or path.is_relative_to(parent)


def collect_raw_files(results_root, out, archive, include=None):
    """Inventory existing inputs without following links or selecting itself."""
    results_root, out, archive = (Path(path).resolve() for path in (results_root, out, archive))
    if not results_root.is_dir():
        raise FileNotFoundError(results_root)
    selected = [results_root] if not include else [
        (Path(path) if Path(path).is_absolute() else results_root/Path(path)).resolve()
        for path in include]
    for path in selected:
        if not _within(path, results_root) or not path.is_dir():
            raise ValueError("included result directories must exist inside results_root")
    files = set()
    for selected_root in selected:
        if _within(selected_root, out):
            continue
        for base, directories, names in os.walk(selected_root, followlinks=False):
            base = Path(base)
            directories[:] = [name for name in directories
                if name not in EXCLUDED_DIRS and not name.startswith(".venv")
                and not name.startswith(("pytest_tmp", "pytest_temp", "pytest_work"))
                and not _within(base/name, out) and not (base/name).is_symlink()
                and not getattr(base/name, "is_junction", lambda: False)()]
            for name in names:
                path = base/name
                if (path == archive or _within(path, out) or path.is_symlink()
                        or path.suffix.lower() in EXCLUDED_SUFFIXES):
                    continue
                files.add(path)
    return sorted(files, key=lambda path: path.relative_to(results_root).as_posix())


def _prediction_statistics(files, results_root):
    by_case, chosen = {}, {}
    for path in files:
        if path.name != "prediction_RMSE_by_horizon.csv":
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            # Preserve older raw output, but do not attach its unequal-domain
            # grouping as the corrected common-support comparison.
            if "common_pplnm_support_scope" not in (reader.fieldnames or []):
                continue
            groups = {}
            for row in reader:
                groups.setdefault(row["case_id"], []).append(dict(row,
                    analysis_path=path.relative_to(results_root).as_posix()))
            metadata_path = path.parent/"summary.json"
            metadata = read_json(metadata_path) if metadata_path.is_file() else {}
            # Prefer a complete cohort over an earlier small subset without
            # doubling the same measured anchor rows. Both raw sources remain.
            rank = (len(metadata.get("cases", [])), path.parent.name)
            for case_id, rows in groups.items():
                if case_id not in chosen or rank > chosen[case_id]:
                    chosen[case_id], by_case[case_id] = rank, rows
    return by_case


def _record_for_case(directory):
    study = directory.parent.parent if directory.parent.name == "cases" else None
    if study is None:
        return {}, {}, 1
    registration_file = study/"EXPERIMENT_CONFIG.json"
    registration = read_json(registration_file) if registration_file.is_file() else {}
    registered = {row["case_id"]: row for row in registration.get("registered_cases", [])}
    records_file = study/"EXPERIMENT_RECORDS.json"
    records = read_json(records_file) if records_file.is_file() else []
    matching = next((row for row in records if row["case_id"] == directory.name), {})
    return registered.get(directory.name, {}), matching, registration.get("workers", 1)


def _no_observed_prediction_targets(directory, planning_cycle_count):
    """Require complete saved planning and observation evidence for no sample.

    Missing/incomplete logs cannot establish that prediction RMSE is undefined.
    An observed target with no completed offline analysis remains pending.
    """
    diagnostics_path = directory/"planning_diagnostics.json"
    observations_path = directory/"observations.json"
    if not diagnostics_path.is_file() or not observations_path.is_file():
        return False
    details, observations = read_json(diagnostics_path), read_json(observations_path)
    if (not isinstance(details, list) or not details
            or planning_cycle_count != len(details)
            or not isinstance(observations, list) or not observations):
        return False
    for detail in details:
        if not isinstance(detail, dict):
            return False
        tracks = detail.get("tracks", detail.get("causal_tracks"))
        if not isinstance(tracks, list) or tracks or detail.get("track_count", 0) != 0:
            return False
    return all(isinstance(packet, dict) and isinstance(packet.get("objects"), list)
               and not packet["objects"] for packet in observations)


def summary_rows(files, results_root):
    """Keep individual configurations separate; never invent an overall rate."""
    predictions = _prediction_statistics(files, results_root)
    rows, configs = [], []
    for path in files:
        if path.name != "summary.json":
            continue
        summary = read_json(path)
        if not isinstance(summary, dict) or summary.get("backend") != "remus6dof":
            continue
        if not all(key in summary for key in ("scenario", "status", "reached_goal")):
            # Response-validation summaries also name the original REMUS
            # backend, but are quality evidence rather than navigation cases.
            continue
        config_path = path.parent/"config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"actual REMUS summary has no matching configuration: {path}")
        cfg = Stage2Config.load(config_path)
        registered, record, workers = _record_for_case(path.parent)
        case_id = path.parent.name
        case = ExperimentCase(case_id, registered.get("group_id", case_id),
            registered.get("suite", "standalone_closed_loop"), summary["scenario"], cfg)
        record = dict(record, summary=summary, summary_path=path.relative_to(results_root).as_posix())
        planning_path = path.parent/"planning.csv"
        if "planning_cycle_count" not in record and planning_path.is_file():
            with planning_path.open(encoding="utf-8") as stream:
                record["planning_cycle_count"] = max(0, sum(bool(line.strip()) for line in stream)-1)
        row = metric_row(case, record, workers)
        relative_case = path.parent.relative_to(results_root).as_posix()
        row["source_case_directory"] = relative_case
        matched = predictions.get(case_id, [])
        if matched:
            row["prediction_RMSE_by_horizon"] = json.dumps(matched, ensure_ascii=False)
            row["prediction_RMSE_status"] = "completed_offline_same_anchor_dt3_common_9s_support; models_explicit_in_cell"
        else:
            row["prediction_RMSE_by_horizon"] = None
            row["prediction_RMSE_status"] = (
                "undefined_no_observed_targets; complete_planning_and_observation_logs; no_measured_zero_implied"
                if _no_observed_prediction_targets(path.parent, record.get("planning_cycle_count"))
                else "pending_offline_same_anchor_evaluation; no_measured_zero_implied")
        rows.append(row)
        configs.append(dict(case_id=case_id, group_id=case.group_id,
            source_case_directory=relative_case, scenario=case.scenario,
            scenario_seed=cfg.seed, config=read_json(config_path)))
    return rows, configs


def build_delivery(results_root, out, archive, *, include=None, documents=None):
    results_root, out, archive = (Path(path).resolve() for path in (results_root, out, archive))
    documents = Path(documents or ROOT/"docs_stage3").resolve()
    if out.exists() or archive.exists():
        raise FileExistsError("delivery directory and ZIP must both be new; existing evidence is preserved")
    if _within(archive, out):
        raise ValueError("ZIP must be outside its own delivery directory")
    if not documents.is_dir() or _within(documents, out):
        raise ValueError("documents must be an existing independent report directory")
    files = collect_raw_files(results_root, out, archive, include)
    rows, configs = summary_rows(files, results_root)
    out.mkdir(parents=True, exist_ok=False)
    for source in files:
        target = out/"raw"/source.relative_to(results_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copytree(documents, out/"reports", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".venv*"))
    for name in ("TEST_REPORT.md", "CHANGELOG.md", "FAILURE_ANALYSIS.md", "PREDICTOR_INTERFACE.md", "ECP_INTERFACES.md"):
        source = documents/name
        if source.is_file():
            shutil.copy2(source, out/name)
    columns = list(dict.fromkeys(key for row in rows for key in row)) or [
        "case_id", "scenario", "scenario_seed", "navigation_success", "collision", "timeout",
        "actual_clearance_margin_violation", "unsafe_emergency_forecast",
        "prediction_RMSE_by_horizon", "prediction_RMSE_status", "actual_close_interactions",
        "planning_P95_ms", "planning_deadline_miss"]
    with (out/"EXPERIMENT_SUMMARY.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(out/"EXPERIMENT_CONFIG.json", dict(cases=configs,
        source_results_root=str(results_root), included_directories=list(include or ["."]),
        common_safety_margin_m=1.0, original_hydrodynamics_modified=False,
        rates_aggregated_across_different_configurations=False))
    explanation = """# Stage3 independent review package

`EXPERIMENT_SUMMARY.csv` contains one row per recorded actual REMUS episode,
with arrival, collision, timeout, actual safety margin and unsafe emergency
reported separately. Different budgets, predictors, terrain and memory settings
are separate configurations; there is no combined success rate. A seed-7-only
comparison is preliminary, not a statistically significant method ranking.

`EXPERIMENT_CONFIG.json` preserves every source configuration and scenario seed.
`reports/` contains repository documentation. `raw/` retains all selected raw
data, including failures, initial/fixed test logs, XML, true state/control/time
histories, causal deliveries, memory events, candidate diagnostics and figures.
`prediction_paired_common_support*` uses corrected common <=9/>9-second support
groups for all predictors. Earlier `prediction_paired_four_cases*` raw output
is preserved as the pre-correction schema, and is not attached as the corrected
group comparison. Missing prediction RMSE remains pending unless complete saved
planning and observation logs establish no observed targets at every planning
cycle; that case is undefined_no_observed_targets. Neither status means zero.
Where a case occurs in several corrected analyses, its metric cell uses the
analysis with the largest completed case cohort, then directory name as a
deterministic tie break. It never duplicates sample rows; every original source
remains under raw/ and every attached metric lists its analysis path.

Offline prediction errors can evaluate different models on the same causal
histories from a CA-driven run. That does not relabel execution as PPLNM-driven
navigation; the executed predictor and the evaluated variants are explicit.
Known-seed future obstacle truth appears only in offline evaluation and original
sensor/collision/assessment logs; it is not an online prediction input.

Planning time is measured wall time. Parallel experiments share CPU resources.
The synchronous physical simulation pauses during planning; a fixed configured
command-delay ablation is distinct from injecting actual measured latency.
The original one-metre margin and collision geometry remain unchanged.

Virtual environments, bytecode, pytest work directories and nested ZIPs are
excluded. Files are copied and compressed normally without any extra digest
or checksum verification. Inventory contains paths and byte sizes only.
"""
    (out/"README_REVIEW.md").write_text(explanation, encoding="utf-8")
    inventory = [{"path": path.relative_to(out).as_posix(), "size_bytes": path.stat().st_size}
                 for path in sorted(out.rglob("*")) if path.is_file()]
    _write_json(out/"FILE_INVENTORY.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        inventory_scope="all delivery files existing before FILE_INVENTORY.json; inventory excludes itself",
        file_count=len(inventory), total_size_bytes=sum(row["size_bytes"] for row in inventory),
        files=inventory))
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive_files = sorted(path for path in out.rglob("*") if path.is_file())
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3, allowZip64=True) as stream:
        for path in archive_files:
            stream.write(path, path.relative_to(out).as_posix())
    # Directory/list/size inspection only, with no content-identity check.
    with zipfile.ZipFile(archive, "r") as stream:
        names = stream.namelist()
    if len(names) != len(archive_files) or len(names) != len(set(names)) or archive.stat().st_size <= 0:
        raise RuntimeError("ZIP file inventory/size does not match the prepared delivery")
    return dict(delivery_directory=str(out), zip_path=str(archive),
                archive_file_count=len(names), archive_size_bytes=archive.stat().st_size,
                actual_remus_episode_count=len(rows), raw_file_count=len(files))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(ROOT/"results_stage3"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--include", nargs="+")
    parser.add_argument("--documents", default=str(ROOT/"docs_stage3"))
    args = parser.parse_args(argv)
    result = build_delivery(args.results_root, args.out, args.zip,
                            include=args.include, documents=args.documents)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
