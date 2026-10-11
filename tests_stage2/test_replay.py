from types import SimpleNamespace
import json
import re

import numpy as np
import pytest

from auv3d.stage2_replay import build_replay_payload, export_stage2_replay, latest_at_or_before
from auv3d.terrain import HeightField


def _result(n=3001):
    states = np.zeros((n, 16))
    states[:, 0] = np.linspace(0, 60, n)
    states[:, 1] = np.linspace(30, 140, n)
    states[:, 2] = 100
    states[:, 3] = 35
    prediction = states[:601].copy()
    prediction[:, 0] -= prediction[0, 0]
    return {"summary": {"backend": "remus6dof", "scenario": "test</script>", "status": "timeout"},
            "states": states, "goal": np.array([210, 100, 35]),
            "terrain": HeightField.flat(260, 200, 90),
            "selected_plans": [{"time_s": 0, "states": prediction, "feasible": True}],
            "obstacles": [{"time_s": 0, "positions": [[100, 130, 35]], "radii": [4], "ids": [8]},
                          {"time_s": 1, "positions": [[100, 129, 35]], "radii": [4], "ids": [8]}],
            "events": [{"time_s": 2, "kind": "emergency", "position": [34, 100, 35], "reason": "coasting"}]}


def test_replay_downsamples_only_display_and_preserves_actual_prediction_endpoints():
    result = _result()
    original = result["states"].copy()
    payload = build_replay_payload(result)
    assert len(payload["actual"]) <= 2400
    np.testing.assert_array_equal(payload["actual"][0], original[0, :7])
    np.testing.assert_array_equal(payload["actual"][-1], original[-1, :7])
    np.testing.assert_array_equal(payload["plans"][0]["states"][-1], result["selected_plans"][0]["states"][-1, :7])
    np.testing.assert_array_equal(result["states"], original)
    assert payload["terrain"]["depths"][0][0] == 90
    assert payload["actual"][0][3] < payload["terrain"]["depths"][0][0]
    assert "positive down" in payload["coordinates"]


def test_replay_uses_latest_past_snapshot_never_future_truth():
    snapshots = _result()["obstacles"]
    assert latest_at_or_before(snapshots, -1) is None
    assert latest_at_or_before(snapshots, .99)["time_s"] == 0
    assert latest_at_or_before(snapshots, 1)["time_s"] == 1
    assert latest_at_or_before(snapshots, 2)["time_s"] == 1


def test_single_html_contains_real_json_without_network_or_markup_injection(tmp_path):
    result = _result()
    result['summary'].update(status='success',safety_validation_status='failed_unsafe_emergency_forecast',
                             avoidance_acceptance_passed=False)
    path = tmp_path / "replay.html"
    export_stage2_replay(result, path)
    html = path.read_text(encoding="utf-8")
    encoded = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    payload = json.loads(encoded)
    assert payload["scenario"] == result["summary"]["scenario"]
    assert payload['status']=='success' and payload['avoidance_acceptance_passed'] is False
    assert payload['safety_validation_status']=='failed_unsafe_emergency_forecast'
    assert 'data.safety_validation_status' in html
    assert payload["actual"][-1][0] == 60
    assert "test</script>" not in encoded
    assert not re.search(r'<script[^>]+src=|<link[^>]+href=|fetch\(|https?://|WebSocket', html)
    assert "depth positive down" in html
    assert "requestAnimationFrame" in html
    with pytest.raises(FileExistsError):
        export_stage2_replay(result, path)


@pytest.mark.parametrize("problem", ["nan", "time", "backend", "prediction", "radius"])
def test_replay_rejects_invalid_or_mislabelled_data(problem):
    result = _result()
    if problem == "nan": result["states"][1, 1] = np.nan
    if problem == "time": result["states"][1, 0] = -1
    if problem == "backend": result["summary"]["backend"] = "p25_kinematic_3d_rk4"
    if problem == "prediction": result["selected_plans"][0]["states"][0, 0] = -1
    if problem == "radius": result["obstacles"][0]["radii"] = [-1]
    with pytest.raises(ValueError):
        build_replay_payload(result)
