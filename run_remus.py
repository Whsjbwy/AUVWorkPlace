"""Reproducible REMUS 100 demo using the unmodified upstream dynamics."""
from pathlib import Path
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from python_vehicle_simulator.vehicles import remus100
from python_vehicle_simulator.lib import simulate, plotControls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=200)
    parser.add_argument("--depth", type=float, default=30)
    parser.add_argument("--heading", type=float, default=50)
    parser.add_argument("--rpm", type=float, default=1525)
    parser.add_argument("--current", type=float, default=0.5)
    parser.add_argument("--current-direction", type=float, default=170)
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(list(vars(args).values())).all() or args.duration <= 0:
        parser.error("Parameters must be finite and duration must be positive")
    out = Path(__file__).resolve().parent / "results" / "remus100"
    out.mkdir(parents=True, exist_ok=True)
    dt = 0.02
    vehicle = remus100("depthHeadingAutopilot", args.depth, args.heading,
                       args.rpm, args.current, args.current_direction)
    started = time.perf_counter()
    print("Running REMUS 100 dynamics...", flush=True)
    times, data = simulate(round(args.duration / dt), dt, vehicle)
    elapsed = time.perf_counter() - started
    if not np.isfinite(data).all():
        raise RuntimeError("Non-finite simulation states")
    t = times.ravel()
    heading = np.rad2deg(np.unwrap(data[:, 5]))
    heading_error = (heading[-1] - args.heading + 180) % 360 - 180
    summary = {
        "parameters": vars(args), "sample_time_s": dt, "samples": len(data),
        "simulation_duration_s": float(t[-1]), "computation_time_s": elapsed,
        "all_states_finite": True, "final_depth_m": float(data[-1, 2]),
        "final_heading_deg": float(heading[-1]),
        "final_depth_error_m": float(data[-1, 2] - args.depth),
        "final_heading_error_deg": float(heading_error),
        "path_length_m": float(np.linalg.norm(np.diff(data[:, :3], axis=0), axis=1).sum()),
    }
    headers = "time_s,north_m,east_m,depth_m,roll_rad,pitch_rad,yaw_rad,u_mps,v_mps,w_mps,p_radps,q_radps,r_radps,rudder_command_rad,stern_command_rad,rpm_command,rudder_actual_rad,stern_actual_rad,rpm_actual"
    np.savetxt(out / "simulation.csv", np.column_stack([t, data]), delimiter=",", header=headers, comments="")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].plot(data[:, 1], data[:, 0])
    axes[0, 0].scatter(data[[0, -1], 1], data[[0, -1], 0], c=["green", "red"])
    axes[0, 0].set(xlabel="East (m)", ylabel="North (m)", title="Horizontal trajectory", aspect="equal")
    axes[0, 1].plot(t, data[:, 2]); axes[0, 1].axhline(args.depth, color="orange", ls="--")
    axes[0, 1].set(title="Depth tracking", ylabel="Depth (m)"); axes[0, 1].invert_yaxis()
    axes[0, 2].plot(t, heading); axes[0, 2].axhline(args.heading, color="orange", ls="--")
    axes[0, 2].set(title="Heading tracking", ylabel="Yaw (deg)")
    axes[1, 0].plot(t, np.linalg.norm(data[:, 6:9], axis=1))
    axes[1, 0].set(title="Body velocity magnitude", ylabel="Speed (m/s)")
    axes[1, 1].plot(t, np.rad2deg(data[:, 3:5])); axes[1, 1].legend(["Roll", "Pitch"])
    axes[1, 1].set(title="Attitude", ylabel="Angle (deg)")
    axes[1, 2].plot(t, data[:, 6:9]); axes[1, 2].legend(["Surge", "Sway", "Heave"])
    axes[1, 2].set(title="Body velocities", ylabel="Velocity (m/s)")
    for i, ax in enumerate(axes.flat):
        ax.grid(alpha=0.3)
        if i: ax.set_xlabel("Time (s)")
    fig.suptitle("REMUS 100 | depth & heading autopilot | 6-DOF dynamics")
    fig.savefig(out / "states.png", dpi=150)
    plotControls(times, data, vehicle, 2)
    plt.figure(2).tight_layout(); plt.figure(2).savefig(out / "controls.png", dpi=150)
    fig3 = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig3.add_subplot(projection="3d")
    east, north, depth = data[:, 1], data[:, 0], data[:, 2]
    ax.plot(east, north, depth, alpha=0.3, label="Full trajectory")
    ax.scatter(east[0], north[0], depth[0], c="green", label="Start")
    ax.scatter(east[-1], north[-1], depth[-1], c="red", label="End")
    ax.set(xlabel="East (m)", ylabel="North (m)", zlabel="Depth (m)", title="REMUS 100 trajectory")
    ax.invert_zaxis(); ax.legend()
    fig3.savefig(out / "trajectory.png", dpi=150)
    if not args.no_animation:
        print("Rendering trajectory animation...", flush=True)
        trail, = ax.plot([], [], [], color="tab:blue", lw=2)
        point, = ax.plot([], [], [], "o", color="orange")
        frames = np.linspace(0, len(data) - 1, 100, dtype=int)
        def update(index):
            trail.set_data_3d(east[:index+1], north[:index+1], depth[:index+1])
            point.set_data_3d([east[index]], [north[index]], [depth[index]])
            ax.set_title(f"REMUS 100 | simulation time {t[index]:.1f} s")
            return trail, point
        animation = FuncAnimation(fig3, update, frames=frames, interval=100)
        animation.save(out / "trajectory.gif", writer=PillowWriter(fps=10), dpi=90)
    plt.close("all")
    print(json.dumps(summary, indent=2))
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
