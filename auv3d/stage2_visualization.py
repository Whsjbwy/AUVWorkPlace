"""Offline diagnostic views of actual REMUS execution and separate predictions.

The figures never feed obstacle truth back into the planner. Stored plan rows
contain relative time followed by REMUS pose, velocity and actuator state; they
must not be interpreted as a second executed trajectory.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .visualization import _draw_terrain, _draw_obstacles, _set_ned_axes, _finish_3d


CONTROL_COLUMNS = (
    "time", "ref_depth_m", "ref_heading_rad", "ref_rpm", "raw_rudder_rad",
    "raw_stern_rad", "raw_rpm", "applied_rudder_rad", "applied_stern_rad",
    "applied_rpm", "actual_rudder_rad", "actual_stern_rad", "actual_rpm",
    "saturated_rudder", "saturated_stern", "saturated_rpm",
)
ERROR_COLUMNS = (
    "time_s", "plan_time_s", "elapsed_s", "position_error_m", "depth_error_m",
    "heading_error_rad", "surge_error_mps",
)


def _data(result):
    states = np.asarray(result["states"], dtype=float)
    goal = np.asarray(result["goal"], dtype=float)
    if states.ndim != 2 or len(states) == 0 or states.shape[1] != 16:
        raise ValueError("REMUS plots require nonempty [time, eta6, nu6, actual3] states")
    if not np.isfinite(states).all() or goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("REMUS plots require finite trajectory and a finite NED goal")
    if np.any(np.diff(states[:, 0]) < 0):
        raise ValueError("REMUS trajectory times must be nondecreasing")
    terrain = result["terrain"]
    if not np.isfinite(terrain.depths).all():
        raise ValueError("terrain depths must be finite")
    for plan in result.get("selected_plans", []):
        rows = np.asarray(plan["states"], dtype=float)
        if rows.ndim != 2 or rows.shape[1] != 16 or not len(rows) or not np.isfinite(rows).all():
            raise ValueError("REMUS planned rollouts require finite [offset, eta6, nu6, actual3] states")
        if np.any(np.diff(rows[:, 0]) < 0) or not np.isfinite(float(plan["time_s"])):
            raise ValueError("REMUS planned rollout times must be finite and nondecreasing")
    return states, goal, terrain


def _table(result, key, column_key, defaults):
    rows = np.asarray(result.get(key, []), dtype=float)
    if rows.size == 0:
        return {}
    names = tuple(result.get(column_key, defaults))
    if rows.ndim != 2 or rows.shape[1] != len(names) or not np.isfinite(rows).all():
        raise ValueError(f"{key} must be a finite table matching its named columns")
    table = {name: rows[:, index] for index, name in enumerate(names)}
    if 'time_s' in table and 'time' not in table:
        table['time'] = table['time_s']
    return table


def _title(result, suffix):
    summary = result.get("summary", {})
    cfg = result.get('config', {})
    method = cfg.get('obstacle_predictor', 'ca').upper()
    if method == 'PPLNM':
        method += ' (trained 9s from measurement + '+cfg.get('pplnm_extension_mode', 'unspecified')+')'
    return (f"{summary.get('scenario', 'scenario')} | navigation: {summary.get('status', 'unreported')} | {suffix}\n"
            f"Safety: {summary.get('safety_validation_status', 'not reported')}\n"
            "REMUS 6-DOF actual execution; separate 6-DOF forward predictions\n"
            +method+' | memory: '+cfg.get('memory_policy','hard_expiry')
            +' | guidance: '+cfg.get('terrain_guidance','direct'))


def _position_at(states, time_s):
    return np.array([np.interp(time_s, states[:, 0], states[:, j]) for j in (1, 2, 3)])


def _draw_actual(ax, states, goal, *, route=True, final=True):
    if route:
        ax.plot(*states[:, 1:4].T, color="#005ca9", linewidth=2.1,
                label="REMUS 6-DOF actual execution")
    ax.scatter(*states[0, 1:4], marker="o", color="#149c47", s=55,
               edgecolors="black", depthshade=False, label="Start")
    ax.scatter(*goal, marker="*", color="#d42c31", s=135,
               edgecolors="black", depthshade=False, label="Goal")
    if route and final:
        ax.scatter(*states[-1, 1:4], color="black", marker="x", s=35, label="Final actual position")


def _draw_predictions(ax, plans, max_plans=5):
    if not plans:
        return
    indices = np.unique(np.linspace(0, len(plans)-1, min(max_plans, len(plans)), dtype=int))
    labels = set()
    for index in indices:
        plan = plans[index]
        rows = np.asarray(plan["states"], dtype=float)
        feasible = bool(plan.get("feasible", True))
        label = "REMUS 6-DOF predicted rollout snapshots" if feasible else "REMUS emergency forward prediction"
        ax.plot(*rows[:, 1:4].T, linestyle="--", linewidth=1.2,
                color="#dd7c00" if feasible else "#bd3028", alpha=.8,
                label=label if label not in labels else None)
        labels.add(label)


def _draw_events(ax, result, states, time_s=np.inf):
    labels = set()
    for event in result.get("events", []):
        stamp = float(event["time_s"])
        if stamp > time_s:
            continue
        kind = str(event.get("kind", "event"))
        position = np.asarray(event.get("position", _position_at(states, stamp)), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("event position must be a finite NED point")
        label = "Emergency action / no feasible candidate" if "emergency" in kind else "Failure / diagnostic event"
        ax.scatter(*position, marker="X", color="#b40c2c", s=65, depthshade=False,
                   label=label if label not in labels else None)
        labels.add(label)


def _limits(states, goal, terrain, local=False):
    if local:
        low = np.minimum(states[:, 1:4].min(0), goal)-15.
        high = np.maximum(states[:, 1:4].max(0), goal)+15.
        low[:2] = np.maximum(low[:2], [terrain.xs[0], terrain.ys[0]])
        high[:2] = np.minimum(high[:2], [terrain.xs[-1], terrain.ys[-1]])
        for j in (0, 1):
            if high[j] <= low[j]:
                low[j], high[j] = (terrain.xs if j == 0 else terrain.ys)[[0, -1]]
        return (low[0], high[0]), (low[1], high[1])
    return (float(terrain.xs[0]), float(terrain.xs[-1])), (float(terrain.ys[0]), float(terrain.ys[-1]))


def _closest_snapshot(states, snapshots):
    best = None
    distance = np.inf
    for snapshot in snapshots:
        positions = np.asarray(snapshot.get("positions", []), dtype=float).reshape(-1, 3)
        if not len(positions):
            continue
        radii = np.asarray(snapshot["radii"], dtype=float)
        separation = float(np.min(np.linalg.norm(positions-_position_at(states, snapshot["time_s"]), axis=1)-radii))
        if separation < distance:
            distance, best = separation, snapshot
    return best


def _plot_environment(result, out, states, goal, terrain):
    snapshots = result.get("obstacles", [])
    for filename, local, route in (("terrain_3d.png", False, False),
                                    ("trajectory_3d.png", False, True),
                                    ("trajectory_local_3d.png", True, True)):
        nlim, elim = _limits(states, goal, terrain, local)
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        surface, bottom = _draw_terrain(ax, terrain, nlim if local else None, elim if local else None)
        snapshot = _closest_snapshot(states, snapshots) if local else (snapshots[-1] if route and snapshots else snapshots[0] if snapshots else None)
        if snapshot is not None:
            _draw_obstacles(ax, snapshot)
        _draw_actual(ax, states, goal, route=route)
        if route:
            _draw_predictions(ax, result.get("selected_plans", []))
            _draw_events(ax, result, states)
        _set_ned_axes(ax, nlim, elim, max(bottom, float(states[:, 3].max()), goal[2]))
        suffix = "local execution and closest obstacle snapshot" if local else "full execution and final obstacle snapshot" if route else "initial environment"
        ax.set_title(_title(result, suffix))
        _finish_3d(fig, ax, surface, out/filename)


def _plot_errors(result, out, errors):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    if errors:
        times = errors["time_s"]
        axes[0].plot(times, errors["position_error_m"], label="Actual minus selected 6-DOF prediction")
        axes[0].set_ylabel("Position error (m)")
        axes[1].plot(times, errors["depth_error_m"], label="Signed depth error")
        axes[1].set_ylabel("Depth error (m)")
        axes[2].plot(times, np.rad2deg(errors["heading_error_rad"]), label="Wrapped heading error")
        axes[2].set_ylabel("Heading error (deg)")
        for ax in axes:
            ax.legend(fontsize=8)
    else:
        axes[0].text(.5, .5, "No prediction comparison samples recorded", ha="center", transform=axes[0].transAxes)
    axes[-1].set_xlabel("Simulation time (s)")
    axes[0].set_title("Active selected-plan vs actual execution error, aligned to original decision time\n"
                      "Includes configured command delay; a new plan is compared only after its command becomes active"
                      if result.get('config',{}).get('command_delay_s',0.) else
                      "Selected-plan prediction error during its actual command interval\nLater replanning is excluded from this error measure")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/"prediction_error.png", dpi=150)
    plt.close(fig)


def _plot_controls(result, out, states, controls):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, name, index in zip(axes, ("rudder", "stern", "rpm"), (13, 14, 15)):
        factor = 1. if name == "rpm" else 180./np.pi
        ax.plot(states[:, 0], states[:, index]*factor, color="#005ca9", label="Actual actuator state")
        suffix = "rpm" if name == "rpm" else name+"_rad"
        if controls:
            times = controls["time"]
            ax.plot(times, controls["raw_"+suffix]*factor, linestyle=":", color="#7d7d7d", label="Raw autopilot request")
            ax.step(times, controls["applied_"+suffix]*factor, where="post", linestyle="--", color="#dd7c00", label="Applied bounded command")
            saturated = controls["saturated_"+name] > 0
            if np.any(saturated):
                ax.scatter(times[saturated], controls["applied_"+suffix][saturated]*factor, s=8,
                           color="#bd3028", label="Request saturation")
        ax.set_ylabel("RPM" if name == "rpm" else name.title()+" (deg)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_title("REMUS autopilot requests, bounded commands and actuator lag")
    axes[-1].set_xlabel("Simulation time (s)")
    fig.tight_layout()
    fig.savefig(out/"controls.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(states[:, 0], states[:, 3], label="REMUS actual depth")
    axes[1].plot(states[:, 0], np.rad2deg(np.unwrap(states[:, 6])), label="REMUS actual heading")
    if controls:
        axes[0].step(controls["time"], controls["ref_depth_m"], where="post", linestyle="--", label="Depth reference")
        axes[1].step(controls["time"], np.rad2deg(np.unwrap(controls["ref_heading_rad"])), where="post", linestyle="--", label="Heading reference")
    axes[0].set_ylabel("Depth (m), positive down")
    axes[0].invert_yaxis()
    axes[1].set_ylabel("Heading (deg, unwrapped)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[0].set_title("Original REMUS depth/heading autopilot response")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"depth_heading.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ground = np.linalg.norm(np.diff(states[:, 1:4], axis=0), axis=1)
    elapsed = np.diff(states[:, 0])
    valid = elapsed > 0
    ax.plot(states[1:, 0][valid], (ground[valid]/elapsed[valid]), label="Actual ground speed from NED displacement")
    ax.plot(states[:, 0], states[:, 7], label="Actual absolute body surge")
    ax.set(xlabel="Simulation time (s)", ylabel="Speed (m/s)", title="REMUS actual speed diagnostics (body surge is not water-relative speed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/"speed.png", dpi=150)
    plt.close(fig)


def _animation_times(states, snapshots, max_frames):
    """Prefer actual truth snapshot times, avoiding invented obstacle motion."""
    stamps = np.array([float(s["time_s"]) for s in snapshots], dtype=float)
    stamps = stamps[(stamps >= states[0, 0]) & (stamps <= states[-1, 0])]
    times = np.unique(stamps) if len(stamps) else states[:, 0]
    indices = np.unique(np.linspace(0, len(times)-1, min(max_frames, len(times)), dtype=int))
    return times[indices]


def _snapshot_at(snapshots, time_s):
    eligible = [snapshot for snapshot in snapshots if float(snapshot["time_s"]) <= time_s+1e-9]
    return max(eligible, key=lambda item: float(item["time_s"])) if eligible else None


def _current_plan(plans, time_s):
    eligible = [plan for plan in plans if float(plan["time_s"]) <= time_s+1e-9]
    return max(eligible, key=lambda item: float(item["time_s"])) if eligible else None


def export_stage2_animation(result, path, *, max_frames=60, fps=8):
    """Write an offline GIF; frames align with recorded obstacle-truth times.

    The time-compressed playback rate is disclosed in the figure. The actual
    vehicle marker is interpolated from physical integration records; obstacle
    snapshots are drawn at their recorded timestamps, never extrapolated.
    """
    if not isinstance(max_frames, int) or max_frames < 2 or fps <= 0:
        raise ValueError("animation requires max_frames>=2 and positive fps")
    states, goal, terrain = _data(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshots = result.get("obstacles", [])
    plans = result.get("selected_plans", [])
    times = _animation_times(states, snapshots, max_frames)
    nlim, elim = _limits(states, goal, terrain, local=True)
    deepest = max(float(np.max(terrain.depths)), float(states[:, 3].max()), goal[2])
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    def draw_frame(index):
        stamp = float(times[index])
        ax.clear()
        _draw_terrain(ax, terrain, nlim, elim)
        before = states[states[:, 0] <= stamp+1e-9]
        _draw_actual(ax, before, goal, final=False)
        position = _position_at(states, stamp)
        ax.scatter(*position, color="#005ca9", edgecolors="black", s=60, label="Actual REMUS at current time")
        snapshot = _snapshot_at(snapshots, stamp)
        if snapshot is not None:
            _draw_obstacles(ax, snapshot)
        plan = _current_plan(plans, stamp)
        if plan is not None:
            rows = np.asarray(plan["states"], dtype=float)
            remaining = rows[rows[:, 0]+float(plan["time_s"]) >= stamp-1e-9]
            if len(remaining):
                _draw_predictions(ax, [{**plan, "states": remaining}], max_plans=1)
        _draw_events(ax, result, states, stamp)
        _set_ned_axes(ax, nlim, elim, deepest)
        ax.set_title(_title(result, f"t={stamp:.2f}s | time-compressed replay"), fontsize=10)
        ax.legend(loc="upper left", fontsize=6)
        fig.tight_layout()
        return []

    animation = FuncAnimation(fig, draw_frame, frames=len(times), blit=False, repeat=False)
    try:
        animation.save(path, writer=PillowWriter(fps=fps), dpi=90)
    finally:
        plt.close(fig)
    return path


def plot_stage2_episode(result, out, *, animate=True, max_frames=60, fps=8):
    """Export actual 6-DOF execution, separate plans and command diagnostics."""
    states, goal, terrain = _data(result)
    controls = _table(result, "controls", "control_columns", CONTROL_COLUMNS)
    errors = _table(result, "prediction_errors", "error_columns", ERROR_COLUMNS)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _plot_environment(result, out, states, goal, terrain)
    _plot_errors(result, out, errors)
    _plot_controls(result, out, states, controls)
    filenames = ("terrain_3d.png", "trajectory_3d.png", "trajectory_local_3d.png",
                 "prediction_error.png", "controls.png", "depth_heading.png", "speed.png")
    paths = {Path(name).stem: str(out/name) for name in filenames}
    if animate:
        paths["animation"] = str(export_stage2_animation(result, out/"trajectory.gif", max_frames=max_frames, fps=fps))
    return paths
