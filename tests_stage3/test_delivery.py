"""Small packaging fixtures protect evidence, without a large archive run."""
import csv
import json
from pathlib import Path
import zipfile

import pytest

from auv3d.stage2_config import Stage2Config
from scripts_stage3.build_delivery import build_delivery, collect_raw_files, summary_rows


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_inventory_excludes_delivery_itself_and_test_work_but_keeps_logs(tmp_path):
    root = tmp_path/"results"
    write(root/"run"/"failed.xml", "<testsuite failures='1'/>")
    write(root/"run"/"failed.log", "one failed test, kept for review")
    write(root/"run"/"pytest_tmp"/"throwaway", "temporary")
    write(root/"run"/"__pycache__"/"unused.pyc", "temporary")
    out = root/"delivery"
    write(out/"raw"/"recursive.txt", "must never contain itself")
    selected = collect_raw_files(root, out, tmp_path/"review.zip")
    assert {path.name for path in selected} == {"failed.xml", "failed.log"}


def test_failed_actual_episode_raw_and_missing_rmse_are_preserved(tmp_path):
    root, docs = tmp_path/"results", tmp_path/"docs"
    case = root/"actual_failure"
    summary = dict(backend="remus6dof", scenario="static3d", seed=7, status="obstacle_collision",
        reached_goal=False, actual_safety_margin_met=False, unsafe_emergency_forecast_count=1,
        avoidance_acceptance_passed=False, minimum_model_clearance_m=-.1, planning_ms_p95=2500.,
        planning_exceeds_control_period_count=1, duration_s=3.)
    write(case/"summary.json", json.dumps(summary))
    write(case/"config.json", json.dumps(Stage2Config().to_dict()))
    write(case/"states.csv", "time_s,north_m\n3,40\n")
    write(case/"first_failure"/"full_grid.json", '{"safe_count":0}')
    pending = root/"actual_pending"
    write(pending/"summary.json", json.dumps(summary))
    write(pending/"config.json", json.dumps(Stage2Config().to_dict()))
    for name, cohort_size, measured_error in (("prediction_subset", 1, 9.), ("prediction_all", 15, 2.5)):
        write(root/name/"summary.json", json.dumps({"cases": [{}]*cohort_size}))
        write(root/name/"prediction_RMSE_by_horizon.csv",
            "case_id,common_pplnm_support_scope,variant,planning_horizon_s,prediction_RMSE_m\n"
            f"actual_failure,within_pplnm_9s_from_measurement,ca_dt3,3.0,{measured_error}\n")
    write(root/"quality"/"pytest.xml", "<testsuite tests='3'/>")
    write(root/"quality"/"responses"/"summary.json", json.dumps(dict(
        task="original_REMUS_physical_response_validation", backend="remus6dof", all_passed=True)))
    write(docs/"TEST_REPORT.md", "Actual failure remains a research result.")
    write(docs/"FAILURE_ANALYSIS.md", "Unsafe coast recorded without reduced thresholds.")
    out, archive = root/"review_delivery", tmp_path/"review.zip"
    result = build_delivery(root, out, archive, documents=docs)
    assert result["actual_remus_episode_count"] == 2
    assert (out/"raw"/"actual_failure"/"states.csv").read_text() == (case/"states.csv").read_text()
    assert (out/"raw"/"actual_failure"/"first_failure"/"full_grid.json").is_file()
    with (out/"EXPERIMENT_SUMMARY.csv").open(encoding="utf-8") as stream:
        rows = {row["case_id"]: row for row in csv.DictReader(stream)}
    row = rows["actual_failure"]
    assert row["navigation_success"] == "False" and row["collision"] == "True"
    assert row["actual_clearance_margin_violation"] == "True"
    attached = json.loads(row["prediction_RMSE_by_horizon"])
    assert len(attached) == 1
    assert float(attached[0]["prediction_RMSE_m"]) == 2.5
    assert attached[0]["analysis_path"].startswith("prediction_all/")
    assert row["prediction_RMSE_status"].startswith("completed")
    assert rows["actual_pending"]["prediction_RMSE_by_horizon"] == ""
    assert rows["actual_pending"]["prediction_RMSE_status"].startswith("pending")
    with zipfile.ZipFile(archive) as stream:
        assert "raw/actual_failure/first_failure/full_grid.json" in stream.namelist()
        assert not any("review_delivery" in name for name in stream.namelist())


def test_package_never_overwrites_existing_evidence_or_archives_itself(tmp_path):
    root, docs = tmp_path/"results", tmp_path/"docs"
    write(root/"evidence.log", "keep")
    write(docs/"TEST_REPORT.md", "keep")
    out, archive = tmp_path/"delivery", tmp_path/"review.zip"
    write(archive, "existing evidence")
    with pytest.raises(FileExistsError):
        build_delivery(root, out, archive, documents=docs)
    assert archive.read_text() == "existing evidence"
    with pytest.raises(ValueError, match="outside"):
        build_delivery(root, out, out/"recursive.zip", documents=docs)
    assert not out.exists()


@pytest.mark.parametrize("evidence,expected", [
    ("complete_empty", "undefined_no_observed_targets"),
    ("missing_diagnostics", "pending"),
    ("empty_diagnostics", "pending"),
    ("missing_track_field", "pending"),
    ("tracked_target", "pending"),
    ("missing_observations", "pending"),
    ("observed_target", "pending"),
    ("partial_diagnostics", "pending"),
])
def test_undefined_rmse_requires_complete_no_target_evidence(tmp_path, evidence, expected):
    case = tmp_path/"actual_episode"
    write(case/"summary.json", json.dumps(dict(backend="remus6dof", scenario="terrain3d",
        status="timeout", reached_goal=False)))
    write(case/"config.json", json.dumps(Stage2Config().to_dict()))
    write(case/"planning.csv", "time_s,tracked_objects\n0,0\n2,0\n")
    details = [{"time_s": 0., "tracks": []}, {"time_s": 2., "tracks": []}]
    observations = [{"measurement_time_s": 0., "delivered_time_s": 0., "objects": []}]
    if evidence == "empty_diagnostics":
        details = []
    elif evidence == "missing_track_field":
        del details[1]["tracks"]
    elif evidence == "tracked_target":
        details[1]["tracks"] = [{"object_id": 1}]
    elif evidence == "observed_target":
        observations[0]["objects"] = [{"object_id": 1}]
    elif evidence == "partial_diagnostics":
        details.pop()
    if evidence != "missing_diagnostics":
        write(case/"planning_diagnostics.json", json.dumps(details))
    if evidence != "missing_observations":
        write(case/"observations.json", json.dumps(observations))
    rows, _ = summary_rows(list(tmp_path.rglob("*json")), tmp_path)
    assert len(rows) == 1
    assert rows[0]["prediction_RMSE_by_horizon"] is None
    assert rows[0]["prediction_RMSE_status"].startswith(expected)
