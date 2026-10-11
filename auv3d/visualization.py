"""Offline plots of the P25 environment and execution, in NED coordinates.

Obstacle truth is only used to draw diagnostics; these plots never feed the
planner. A stored DWA rollout is a prediction, not a REMUS execution trace.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def _episode_data(result):
    states = np.asarray(result["states"], dtype=float)
    goal = np.asarray(result["goal"], dtype=float)
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] < 7:
        raise ValueError("plotting requires nonempty [time, N, E, D, u, pitch, yaw] states")
    if not np.isfinite(states).all() or goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("plotting requires finite trajectory and a finite NED goal")
    if np.any(np.diff(states[:, 0]) < 0):
        raise ValueError("trajectory times must be nondecreasing")
    terrain = result["terrain"]
    if not np.isfinite(terrain.depths).all():
        raise ValueError("terrain depths must be finite")
    return states, goal, terrain


def _set_ned_axes(ax, north_limits, east_limits, deepest):
    """Set limits explicitly: surface at the top, greater depth at the bottom."""
    ax.set_xlim(*north_limits)
    ax.set_ylim(*east_limits)
    ax.set_zlim(max(float(deepest), 1.), 0.)
    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_zlabel("Depth (m), NED positive down")
    ax.view_init(elev=25, azim=-60)
    ax.set_box_aspect((1.3, 1., .75))


def _draw_terrain(ax, terrain, north_limits=None, east_limits=None):
    if north_limits is None:
        xs, ys = np.meshgrid(terrain.xs, terrain.ys)
        depths = terrain.depths
    else:
        xs, ys = np.meshgrid(np.linspace(*north_limits, 40), np.linspace(*east_limits, 40))
        depths = terrain.depth_at(np.stack((xs, ys), axis=-1))
    surface = ax.plot_surface(xs, ys, depths, cmap="terrain", alpha=.45,
                              linewidth=0, antialiased=True, rcount=50, ccount=50)
    return surface, float(np.max(depths))


def _draw_markers(ax, states, goal, draw_route=True):
    if draw_route:
        ax.plot(states[:, 1], states[:, 2], states[:, 3], color="#005ca9",
                linewidth=2.2, label="P25 executed trajectory")
        ax.scatter(*states[-1, 1:4], marker="x", color="black", s=40, label="Final position")
    ax.scatter(*states[0, 1:4], marker="o", color="#149c47", edgecolors="black",
               s=60, depthshade=False, label="Start")
    ax.scatter(*goal, marker="*", color="#d42c31", edgecolors="black",
               s=150, depthshade=False, label="Goal")


def _draw_obstacles(ax, snapshot, limits=None):
    positions = np.asarray(snapshot.get("positions", []), dtype=float).reshape(-1, 3)
    radii = np.asarray(snapshot.get("radii", []), dtype=float)
    if len(positions) != len(radii) or not np.isfinite(positions).all() or not np.isfinite(radii).all():
        raise ValueError("obstacle positions/radii must be finite and have equal lengths")
    if np.any(radii <= 0):
        raise ValueError("obstacle radii must be positive")
    ids = snapshot.get("ids", list(range(len(positions))))
    visible = np.ones(len(positions), dtype=bool)
    if limits is not None:
        low, high = limits
        visible = np.all(positions + radii[:, None] >= low, axis=1)
        visible &= np.all(positions - radii[:, None] <= high, axis=1)
    positions, radii = positions[visible], radii[visible]
    selected_ids = np.asarray(ids)[visible].tolist()
    if len(positions) == 0:
        return selected_ids
    stamp = float(snapshot.get("time_s", 0.))
    if len(positions) > 12:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   s=np.clip(radii ** 2 * 1.3, 12, 110), color="#a92579", alpha=.65,
                   label=f"Obstacle centers at t={stamp:g}s (marker size indicates radius)")
    else:
        azimuth, polar = np.linspace(0, 2*np.pi, 16), np.linspace(0, np.pi, 10)
        for index, (position, radius) in enumerate(zip(positions, radii)):
            x = position[0] + radius*np.outer(np.cos(azimuth), np.sin(polar))
            y = position[1] + radius*np.outer(np.sin(azimuth), np.sin(polar))
            z = position[2] + radius*np.outer(np.ones_like(azimuth), np.cos(polar))
            ax.plot_wireframe(x, y, z, color="#a92579", alpha=.65,
                              label=f"Obstacle spheres at t={stamp:g}s" if index == 0 else None)
    return selected_ids


def _draw_plans(ax, selected_plans, max_plans=5):
    """Show sparse, separate rollout snapshots; never splice futures into a route."""
    if not selected_plans:
        return False
    indices = np.unique(np.linspace(0, len(selected_plans)-1, min(max_plans, len(selected_plans)), dtype=int))
    labelled = set()
    for index in indices:
        plan = selected_plans[index]
        states = np.asarray(plan["states"], dtype=float)
        if states.ndim != 2 or states.shape[1] < 3 or not np.isfinite(states).all():
            raise ValueError("stored planned rollouts require finite [N, E, D, ...] states")
        feasible = bool(plan.get("feasible", True))
        label = "P25 planned rollout snapshots" if feasible else "P25 emergency fallback rollout"
        ax.plot(states[:, 0], states[:, 1], states[:, 2], linestyle="--",
                color="#dd7c00" if feasible else "#bd3028", alpha=.75, linewidth=1.1,
                label=label if feasible not in labelled else None)
        labelled.add(feasible)
    return True


def _title(result, suffix):
    summary = result["summary"]
    return (f"{summary['scenario']} | {summary['status']} | {suffix}\n"
            "P25 3D kinematics; not REMUS 6-DOF avoidance")


def _finish_3d(fig, ax, surface, path):
    seabed = Line2D([], [], color="#8d9671", linewidth=7, alpha=.6, label="Seabed (depth field)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [seabed], labels + [seabed.get_label()], loc="upper left", fontsize=7)
    fig.colorbar(surface, ax=ax, shrink=.5, pad=.10, label="Seabed depth (m), positive down")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_episode(result, out):
    """Save terrain, global/local trajectories, depth and speed diagnostics."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    states, goal, terrain = _episode_data(result)
    snapshots = result.get("obstacles", [])
    plans = result.get("selected_plans", [])
    north_limits = (float(terrain.xs[0]), float(terrain.xs[-1]))
    east_limits = (float(terrain.ys[0]), float(terrain.ys[-1]))
    deepest = max(float(np.max(terrain.depths)), float(np.max(states[:, 3])), goal[2])
    for name, route, suffix in (("terrain_3d.png", False, "initial environment"),
                                 ("trajectory_3d.png", True, "full environment")):
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        surface, _ = _draw_terrain(ax, terrain)
        if snapshots:
            _draw_obstacles(ax, snapshots[0])
        _draw_markers(ax, states, goal, draw_route=route)
        if route:
            _draw_plans(ax, plans)
        _set_ned_axes(ax, north_limits, east_limits, deepest)
        ax.set_title(_title(result, suffix))
        _finish_3d(fig, ax, surface, out/name)
    plot_local_view(result, out)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(states[:, 0], states[:, 3], label="P25 executed depth")
    ax.axhline(goal[2], linestyle="--", label="Goal depth")
    ax.set(xlabel="Time (s)", ylabel="Depth (m), positive down", title="P25 depth history")
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"depth.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(states[:, 0], states[:, 4], label="P25 surge state")
    ax.set(xlabel="Time (s)", ylabel="Surge speed (m/s)", title="P25 kinematic speed; not a REMUS speed response")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"speed.png", dpi=150)
    plt.close(fig)


def plot_local_view(result, out):
    """Draw local seabed and truth at the closest recorded obstacle approach."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    states, goal, terrain = _episode_data(result)
    snapshots = result.get("obstacles", [])
    low = np.minimum(states[:, 1:4].min(0), goal)-15.
    high = np.maximum(states[:, 1:4].max(0), goal)+15.
    low[:2] = np.maximum(low[:2], [terrain.xs[0], terrain.ys[0]])
    high[:2] = np.minimum(high[:2], [terrain.xs[-1], terrain.ys[-1]])
    low[2] = 0.
    best, best_distance = None, np.inf
    for snapshot in snapshots:
        positions = np.asarray(snapshot["positions"], dtype=float).reshape(-1, 3)
        if not len(positions):
            continue
        auv = np.array([np.interp(snapshot["time_s"], states[:, 0], states[:, j]) for j in (1, 2, 3)])
        distance = float(np.min(np.linalg.norm(positions-auv, axis=1)-np.asarray(snapshot["radii"])))
        if distance < best_distance:
            best, best_distance = snapshot, distance
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface, bottom_depth = _draw_terrain(ax, terrain, (low[0], high[0]), (low[1], high[1]))
    high[2] = max(high[2], bottom_depth)
    _draw_markers(ax, states, goal)
    has_plans = _draw_plans(ax, result.get("selected_plans", []))
    if best is not None:
        selected_ids = _draw_obstacles(ax, best, (low, high))
        if len(selected_ids) <= 5:
            first_motion = True
            for obstacle_id in selected_ids:
                path = np.asarray([s["positions"][s["ids"].index(obstacle_id)]
                                   for s in snapshots if obstacle_id in s["ids"]], dtype=float)
                if len(path) < 2 or np.allclose(path, path[0]):
                    continue
                # Preserve continuity and let the axes clip the motion at the view boundary.
                ax.plot(path[:, 0], path[:, 1], path[:, 2], linestyle=":", color="#a92579", alpha=.8,
                        label="Recorded obstacle motion (offline truth)" if first_motion else None)
                first_motion = False
    _set_ned_axes(ax, (low[0], high[0]), (low[1], high[1]), high[2])
    ax.set_title(_title(result, "local route and seabed"))
    if not has_plans:
        ax.text2D(.02, .01, "No planned rollout snapshots stored in this result.", transform=ax.transAxes, fontsize=8)
    _finish_3d(fig, ax, surface, out/"trajectory_local_3d.png")
