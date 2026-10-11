"""REMUS render regressions: 6-DOF labels, NED depth and synchronized replay."""
import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from auv3d.terrain import HeightField
from auv3d.stage2_visualization import (
    CONTROL_COLUMNS, ERROR_COLUMNS, _animation_times, _snapshot_at, _current_plan,
    export_stage2_animation, plot_stage2_episode,
)


def example_result():
    states = np.zeros((3, 16))
    states[:, 0] = [0., 1., 2.]
    states[:, 1:4] = [[30., 40., 25.], [32., 41., 25.2], [34., 42., 25.4]]
    states[:, 6] = [.1, .12, .14]
    states[:, 7] = [1., 1.2, 1.4]
    states[:, 13:16] = [[0., 0., 800.], [.01, .02, 850.], [.02, .03, 900.]]
    controls = np.zeros((3, 16))
    controls[:, 0] = states[:, 0]
    controls[:, 1:4] = [27., .3, 1000.]
    controls[:, 4:7] = [.7, -.7, 1600.]
    controls[:, 7:10] = [.5, -.5, 1525.]
    controls[:, 10:13] = states[:, 13:16]
    controls[:, 13:16] = 1.
    errors = np.array([[.5, 0., .5, .002, -.001, .0001, -.0002],
                       [1., 0., 1., .003, -.002, .0002, -.0003]])
    return {
        "states": states,
        "goal": np.array([50., 50., 27.]),
        "terrain": HeightField(np.array([0., 100.]), np.array([0., 100.]),
                               np.array([[90., 85.], [85., 80.]]), .1),
        "summary": {"scenario": "crossing3d", "status": "timeout", "backend": "remus6dof"},
        "obstacles": [{"time_s": t, "ids": [0], "positions": [[40., 45.-t, 26.]], "radii": [2.]}
                      for t in (0., 1., 2.)],
        "selected_plans": [{"time_s": 0., "feasible": True, "states": states.copy(),
                            "action": [27., .3, 1000.]}],
        "controls": controls, "control_columns": CONTROL_COLUMNS,
        "prediction_errors": errors, "error_columns": ERROR_COLUMNS,
        "events": [{"time_s": 1., "kind": "emergency", "position": states[1, 1:4].tolist(),
                    "reason": "No feasible candidate; feasible braking is not guaranteed."}],
    }


def test_actual_and_predicted_paths_remain_separate_with_ned_exports(tmp_path, monkeypatch):
    captured = {}
    original = Figure.savefig

    def capture(fig, path, *args, **kwargs):
        captured[path.name] = fig
        original(fig, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    result = example_result()
    result['summary'].update(status='success',safety_validation_status='failed_unsafe_emergency_forecast',
                             avoidance_acceptance_passed=False)
    paths = plot_stage2_episode(result, tmp_path, animate=False)
    assert set(captured) == {"terrain_3d.png", "trajectory_3d.png", "trajectory_local_3d.png",
                             "prediction_error.png", "controls.png", "depth_heading.png", "speed.png"}
    assert len(paths) == 7
    assert all((tmp_path/name).stat().st_size > 1000 for name in captured)
    for name in ("terrain_3d.png", "trajectory_3d.png", "trajectory_local_3d.png"):
        ax = captured[name].axes[0]
        bottom, surface = ax.get_zlim()
        assert surface == 0. and bottom >= 80.
        assert "REMUS 6-DOF actual execution" in ax.get_title()
        assert "navigation: success" in ax.get_title()
        assert "Safety: failed_unsafe_emergency_forecast" in ax.get_title()
        labels = ax.get_legend_handles_labels()[1]
        assert "Start" in labels and "Goal" in labels
        assert not any("P25" in label for label in labels)
    ax = captured["trajectory_local_3d.png"].axes[0]
    lines = {line.get_label(): line for line in ax.lines}
    actual = lines["REMUS 6-DOF actual execution"]
    predicted = lines["REMUS 6-DOF predicted rollout snapshots"]
    np.testing.assert_allclose(np.column_stack(actual.get_data_3d()), result["states"][:, 1:4])
    np.testing.assert_allclose(np.column_stack(predicted.get_data_3d()), result["selected_plans"][0]["states"][:, 1:4])
    assert actual.get_linestyle() == "-" and predicted.get_linestyle() == "--"
    assert "Emergency action / no feasible candidate" in ax.get_legend_handles_labels()[1]
    control_labels = captured["controls.png"].axes[0].get_legend_handles_labels()[1]
    assert {"Actual actuator state", "Raw autopilot request", "Applied bounded command", "Request saturation"} <= set(control_labels)
    assert captured["depth_heading.png"].axes[0].yaxis_inverted()


def test_animation_selects_recorded_truth_times_and_current_plan():
    result = example_result()
    np.testing.assert_array_equal(_animation_times(result["states"], result["obstacles"], 2), [0., 2.])
    assert _snapshot_at(result["obstacles"], 1.4)["time_s"] == 1.
    assert _snapshot_at(result["obstacles"], -.1) is None
    future = {**result["selected_plans"][0], "time_s": 2.}
    assert _current_plan(result["selected_plans"]+[future], 1.)["time_s"] == 0.
    assert _current_plan(result["selected_plans"]+[future], 2.)["time_s"] == 2.


def test_gif_animation_is_actually_exported_with_distinct_frames(tmp_path):
    path = export_stage2_animation(example_result(), tmp_path/"trajectory.gif", max_frames=3, fps=4)
    assert path.stat().st_size > 1000
    with Image.open(path) as gif:
        assert gif.n_frames == 3
        assert gif.info["duration"] == 250
        assert gif.size[0] >= 600 and gif.size[1] >= 400


@pytest.mark.parametrize("case", ["actual", "planned", "controls", "errors"])
def test_invalid_numeric_records_are_rejected_before_any_export(tmp_path, case):
    result = example_result()
    if case == "actual":
        result["states"][1, 3] = np.nan
    elif case == "planned":
        result["selected_plans"][0]["states"][1, 3] = np.nan
    elif case == "controls":
        result["controls"][1, 4] = np.nan
    else:
        result["prediction_errors"][1, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        plot_stage2_episode(result, tmp_path, animate=False)
    assert not list(tmp_path.iterdir())
