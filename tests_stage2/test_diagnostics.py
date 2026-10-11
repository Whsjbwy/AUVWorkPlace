import json

import numpy as np
import pytest

from auv3d.stage2_diagnostics import RESPONSE_COLUMNS, run_responses, run_tracking_diagnostic


def test_real_response_validation_and_saved_traces(tmp_path):
    out = tmp_path / "responses"
    result = run_responses(out)
    assert result["all_passed"]
    assert result["case_count"] == 9
    assert result["hydrodynamic_parameters_modified"] is False
    assert result["obstacle_avoidance_claim"] is False
    cases = {row["case"]: row for row in result["cases"]}
    assert cases["rpm_600_to_1400"]["speed_increased"]
    assert abs(cases["depth_35_to_40"]["final_depth_error_m"]) < 1
    assert abs(cases["heading_0_to_40deg"]["final_heading_error_deg"]) < 2
    currents = [row for row in result["cases"] if row["case"].startswith("current_")]
    assert len(currents) == 6
    assert {0, 90, 180, 270}.issubset({row["current_direction_deg"] for row in currents})
    for row in result["cases"]:
        data = np.loadtxt(out / f"{row['case']}.csv", delimiter=",", skiprows=1)
        assert data.shape[1] == len(RESPONSE_COLUMNS)
        assert np.isfinite(data).all()
        assert data[-1, 0] == row["duration_s"]
        np.testing.assert_allclose(np.diff(data[:, 0]), .02, atol=1e-12)
        assert row["maximum_current_frame_identity_residual_mps"] < 1e-10
        assert row["actuator_state_within_original_limits"]
    assert (out / "control_responses.png").stat().st_size > 1000
    assert (out / "horizontal_current_responses.png").stat().st_size > 1000
    stored = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert stored["all_passed"] is True


def test_real_diagnostic_distinguishes_exact_prediction_and_ideal_kinematics(tmp_path):
    out = tmp_path / "diagnostic"
    result = run_tracking_diagnostic(out)
    assert result["passed"]
    assert result["exact_prediction_matches_actual"]
    assert result["obstacle_avoidance_claim"] is False
    assert result["p25_ideal_rate_position_final_error_m"] > 1
    actual = np.loadtxt(out / "actual_remus.csv", delimiter=",", skiprows=1)
    predicted = np.loadtxt(out / "predicted_remus.csv", delimiter=",", skiprows=1)
    nominal = np.loadtxt(out / "ideal_p25.csv", delimiter=",", skiprows=1)
    errors = np.loadtxt(out / "prediction_errors.csv", delimiter=",", skiprows=1)
    assert actual.shape == predicted.shape == (401, 16)
    np.testing.assert_allclose(actual, predicted, atol=1e-8, rtol=0)
    np.testing.assert_allclose(actual[0, 1:4], nominal[0, 1:4])
    assert errors[:, 1].max() < 1e-8
    assert errors[-1, 2] == pytest.approx(result["p25_ideal_rate_position_final_error_m"])
    assert (out / "tracking_diagnostic.png").stat().st_size > 1000


@pytest.mark.parametrize("runner", [run_responses, run_tracking_diagnostic])
def test_diagnostics_never_overwrite_previous_results(tmp_path, runner):
    out = tmp_path / "existing"
    out.mkdir()
    original = out / "important_previous_run.txt"
    original.write_text("keep this experiment", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner(out)
    assert original.read_text(encoding="utf-8") == "keep this experiment"
    assert list(out.iterdir()) == [original]
