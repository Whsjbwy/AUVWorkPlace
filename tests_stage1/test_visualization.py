"""Regression checks for NED rendering and honest P25 trajectory labels."""
import numpy as np
import pytest
from matplotlib.figure import Figure
from auv3d.terrain import HeightField
from auv3d.visualization import plot_episode, plot_local_view


def example_result():
    return {
        "states": np.array([[0., 60., 80., 35., 1., 0., 0.],
                            [1., 65., 82., 36., 1., 0., 0.],
                            [2., 70., 84., 37., 1., 0., 0.]]),
        "goal": np.array([90., 90., 40.]),
        "terrain": HeightField(np.array([0., 100.]), np.array([0., 100.]),
                               np.array([[100., 95.], [95., 90.]]), .1),
        "summary": {"scenario": "crossing3d", "status": "timeout"},
        "obstacles": [{"time_s": time, "ids": [0], "positions": [[75., 90.-time, 38.]], "radii": [3.]}
                      for time in (0., 1., 2.)],
        "selected_plans": [{"time_s": 0., "feasible": True,
                            "states": [[60., 80., 35., 1., 0., 0.],
                                       [67., 81., 34., 1., .02, .01]]}],
    }


def test_ned_environment_and_planned_vs_executed_exports(tmp_path, monkeypatch):
    captured = {}
    original_savefig = Figure.savefig

    def record_figure(fig, path, *args, **kwargs):
        captured[path.name] = fig
        original_savefig(fig, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record_figure)
    result = example_result()
    plot_episode(result, tmp_path)
    expected = {"terrain_3d.png", "trajectory_3d.png", "trajectory_local_3d.png", "depth.png", "speed.png"}
    assert set(captured) == expected
    for name in expected:
        assert (tmp_path/name).stat().st_size > 1000
    for name in ("terrain_3d.png", "trajectory_3d.png", "trajectory_local_3d.png"):
        ax = captured[name].axes[0]
        bottom, surface = ax.get_zlim()
        assert surface == 0.
        assert bottom >= 90.  # The seabed is included, below the shallower route.
        assert bottom > surface
        assert "not REMUS 6-DOF avoidance" in ax.get_title()
        labels = ax.get_legend_handles_labels()[1]
        assert "Start" in labels and "Goal" in labels
    ax = captured["trajectory_local_3d.png"].axes[0]
    lines = {line.get_label(): line for line in ax.lines}
    executed = lines["P25 executed trajectory"]
    planned = lines["P25 planned rollout snapshots"]
    np.testing.assert_allclose(np.column_stack(executed.get_data_3d()), result["states"][:, 1:4])
    np.testing.assert_allclose(np.column_stack(planned.get_data_3d()), np.asarray(result["selected_plans"][0]["states"])[:, :3])
    assert planned.get_linestyle() == "--"


def test_legacy_result_does_not_invent_a_planned_trajectory(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *args, **kwargs: captured.append(fig))
    result = example_result()
    result.pop("selected_plans")
    plot_local_view(result, tmp_path)
    ax = captured[0].axes[0]
    assert "P25 planned rollout snapshots" not in ax.get_legend_handles_labels()[1]
    assert any("No planned rollout" in text.get_text() for text in ax.texts)


def test_nonfinite_trajectory_is_rejected_before_rendering(tmp_path):
    result = example_result()
    result["states"][1, 3] = np.nan
    with pytest.raises(ValueError, match="finite trajectory"):
        plot_episode(result, tmp_path)
    assert not list(tmp_path.iterdir())
