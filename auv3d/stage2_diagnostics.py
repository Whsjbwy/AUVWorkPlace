"""Physical response checks and an explicitly open-water tracking diagnostic.

All true vehicle runs use the retained REMUS model and original autopilot.  The
P25 comparison is labelled a kinematic command assumption, not REMUS execution
and not an obstacle-avoidance result.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .adapters.remus_guidance import RemusGuidance, ReferenceAction, predict_reference, velocity_frames
from .geometry import kinematic_step, wrap_pi
from .remus import RemusPlant


DT = 0.02
RESPONSE_COLUMNS = [
    "time_s", "north_m", "east_m", "depth_m", "roll_rad", "pitch_rad", "yaw_rad",
    "u_mps", "v_mps", "w_mps", "p_radps", "q_radps", "r_radps",
    "actual_rudder_rad", "actual_stern_rad", "actual_rpm",
    "reference_depth_m", "reference_heading_rad", "reference_rpm",
    "ground_north_mps", "ground_east_mps", "ground_down_mps",
    "water_relative_surge_geometric_mps", "water_relative_sway_geometric_mps", "water_relative_heave_geometric_mps",
    "raw_rudder_rad", "raw_stern_rad", "raw_rpm", "saturated_rudder", "saturated_stern", "saturated_rpm",
]


def _new_output(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; existing experiment results are preserved")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _new_plant(rpm=1100.0, current=0.0, direction=90.0):
    return RemusPlant(depth=35.0, heading_deg=0.0, rpm=rpm,
                      current_speed=current, current_direction_deg=direction,
                      eta0=np.array([30.0, 100.0, 35.0, 0.0, 0.0, 0.0]),
                      initialize_references=True)


def _response(case, initial, target, duration=100.0, change_time=20.0, current=0.0, direction=90.0):
    plant = _new_plant(initial.rpm, current, direction)
    rows = []
    samples = []
    max_frame_residual = 0.0
    for i in range(round(duration / DT) + 1):
        t = i * DT
        action = initial if t < change_time else target
        frames = velocity_frames(plant)
        rotation = plant.gnc.Rzyx(*plant.eta[3:])
        residual = frames["ground_ned_mps"] - (rotation @ frames["relative_body_geometric_mps"] + frames["current_ned_mps"])
        max_frame_residual = max(max_frame_residual, float(np.max(np.abs(residual))))
        # Raw actuator request is recorded for the interval immediately starting
        # at this row. The final row has no subsequent interval.
        row = [t, *plant.eta, *plant.nu, *plant.actual, *action.as_array(),
               *frames["ground_ned_mps"], *frames["relative_body_geometric_mps"]]
        if i < round(duration / DT):
            control = RemusGuidance.step(plant, action, DT)
            row += [*control.raw, *control.saturated.astype(float)]
            samples.append(control)
        else:
            row += [0.0] * 6
        rows.append(row)
    data = np.asarray(rows)
    before = (data[:, 0] >= change_time - 5) & (data[:, 0] < change_time)
    tail = data[:, 0] >= duration - 10
    limit = np.array([plant.vehicle.deltaMax_r, plant.vehicle.deltaMax_s, plant.vehicle.nMax])
    actuator_ok = bool(np.all(abs(data[:, 13:16]) <= limit + 1e-10))
    requested_ok = all(np.all(abs(sample.applied) <= limit + 1e-10) for sample in samples)
    summary = {
        "case": case, "backend": "original_remus6dof", "duration_s": duration,
        "reference_change_time_s": change_time, "initial_reference": initial.as_array().tolist(),
        "target_reference": target.as_array().tolist(), "current_speed_mps": current,
        "current_direction_deg": direction, "all_finite": bool(np.isfinite(data).all()),
        "initial_speed_mean_last5s_before_step_mps": float(data[before, 22].mean()),
        "final_relative_speed_mean_last10s_mps": float(data[tail, 22].mean()),
        "final_ground_ned_mean_last10s_mps": data[tail, 19:22].mean(axis=0).tolist(),
        "final_depth_error_m": float(plant.eta[2] - target.depth_m),
        "final_heading_error_deg": float(np.rad2deg(wrap_pi(plant.eta[5] - target.heading_rad))),
        "actuator_state_within_original_limits": actuator_ok,
        "applied_commands_within_original_limits": requested_ok,
        "maximum_current_frame_identity_residual_mps": max_frame_residual,
        "saturated_control_interval_count": sum(bool(np.any(sample.saturated)) for sample in samples),
        "physics_dt_s": DT, "started_from_rest": True,
        "hydrodynamic_parameters_modified": False,
    }
    summary["passed"] = bool(summary["all_finite"] and actuator_ok and requested_ok
                             and abs(summary["final_depth_error_m"]) < 1.0
                             and abs(summary["final_heading_error_deg"]) < 2.0
                             and max_frame_residual < 1e-10)
    if case == "rpm_600_to_1400":
        summary["speed_increased"] = summary["final_relative_speed_mean_last10s_mps"] > summary["initial_speed_mean_last5s_before_step_mps"] + 0.3
        summary["passed"] = summary["passed"] and summary["speed_increased"]
    return summary, data


def run_responses(out):
    """Run from-rest speed/turn/depth/flow tests and save actual state traces."""
    out = _new_output(out)
    tic = time.perf_counter()
    cruise = ReferenceAction(35.0, 0.0, 1100.0)
    cases = [
        ("rpm_600_to_1400", ReferenceAction(35, 0, 600), ReferenceAction(35, 0, 1400), 100., 0., 90.),
        ("depth_35_to_40", cruise, ReferenceAction(40, 0, 1100), 100., 0., 90.),
        ("heading_0_to_40deg", cruise, ReferenceAction(35, np.deg2rad(40), 1100), 100., 0., 90.),
    ]
    cases += [(f"current_{label}_{speed:g}mps", cruise, cruise, 80., speed, direction)
              for label, direction, speed in (("north", 0, .3), ("east", 90, .3),
                                             ("south", 180, .3), ("west", 270, .3),
                                             ("northeast", 45, .1), ("southwest", 225, .5))]
    results, series = [], {}
    for name, initial, target, duration, current, direction in cases:
        result, data = _response(name, initial, target, duration=duration, current=current, direction=direction)
        results.append(result)
        series[name] = data
        np.savetxt(out / f"{name}.csv", data, delimiter=",", header=",".join(RESPONSE_COLUMNS), comments="")
    summary = {
        "task": "original_REMUS_physical_response_validation", "backend": "remus6dof",
        "all_passed": all(result["passed"] for result in results), "case_count": len(results),
        "physics_dt_s": DT, "hydrodynamic_parameters_modified": False,
        "position_integration": "Rzyx @ absolute_body_nu; current is not added again",
        "current_model_note": "original dynamics use yaw-only body-current approximation; full 3D geometric relative velocity is logged separately",
        "acceptance": "all finite; original actuator limits; final depth error<1m; final heading error<2deg; speed step increases speed>0.3m/s",
        "wall_clock_s": time.perf_counter() - tic, "cases": results,
        "obstacle_avoidance_claim": False,
    }
    _json(out / "summary.json", summary)
    _plot_responses(series, out)
    return summary


def _plot_responses(series, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    data = series["rpm_600_to_1400"]
    axes[0, 0].plot(data[:, 0], data[:, 22], label="Actual water-relative surge")
    axes[0, 0].set(ylabel="Speed (m/s)", title="REMUS speed response: RPM 600 to 1400")
    data = series["depth_35_to_40"]
    axes[0, 1].plot(data[:, 0], data[:, 3], label="REMUS actual depth")
    axes[0, 1].plot(data[:, 0], data[:, 16], "--", label="Depth reference")
    axes[0, 1].set(ylabel="NED depth (m, down +)", title="Depth reference tracking")
    axes[0, 1].invert_yaxis()
    data = series["heading_0_to_40deg"]
    axes[1, 0].plot(data[:, 0], np.rad2deg(data[:, 6]), label="REMUS actual heading")
    axes[1, 0].plot(data[:, 0], np.rad2deg(data[:, 17]), "--", label="Heading reference")
    axes[1, 0].set(ylabel="Heading (deg)", title="Original rate-limited reference dynamics")
    data = series["rpm_600_to_1400"]
    axes[1, 1].plot(data[:, 0], data[:, 15], label="Actual propeller RPM")
    axes[1, 1].plot(data[:, 0], data[:, 18], "--", label="RPM reference")
    axes[1, 1].set(ylabel="RPM", title="Original propeller actuator lag")
    for ax in axes.ravel():
        ax.set_xlabel("Simulation time (s)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(out / "control_responses.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for name, data in series.items():
        if name.startswith("current_"):
            ax.plot(data[:, 2], data[:, 1], label=name.replace("current_", ""))
    ax.set(xlabel="East (m)", ylabel="North (m)", title="REMUS actual ground tracks under horizontal currents")
    ax.axis("equal")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(out / "horizontal_current_responses.png", dpi=140)
    plt.close(fig)


def run_tracking_diagnostic(out):
    """Compare isolated exact prediction with execution and an ideal-rate path.

    This is open water. A large target change remains a reference change: the
    real plant must retain autopilot and actuator lag. The P25 command below is
    a separate, explicitly idealized [u,q,r] assumption, never a fin command.
    """
    out = _new_output(out)
    tic = time.perf_counter()
    plant = _new_plant()
    initial = ReferenceAction(35.0, 0.0, 1100.0)
    for _ in range(round(20.0 / DT)):
        RemusGuidance.step(plant, initial, DT)
    plant.eta[:3] = [30.0, 100.0, 35.0]
    plant.time_s = 0.0
    action = ReferenceAction(float(plant.eta[2] + 8), float(plant.eta[5] + np.deg2rad(70)), 1400.0)
    horizon = 8.0
    prediction = predict_reference(plant, action, horizon, DT)
    kinematic_state = np.array([*plant.eta[:3], max(0., plant.relative_velocity()[0]), plant.eta[4], plant.eta[5]])
    # Within stage 1's development q/r envelope. This model imposes those rates
    # and the new speed immediately; REMUS's reference action cannot do so.
    p25_command = np.array([2.5 * action.rpm / plant.vehicle.nMax, -np.deg2rad(4), np.deg2rad(5)])
    states = []
    p25_states = []
    controls = []
    for i in range(round(horizon / DT) + 1):
        t = i * DT
        states.append([t, *plant.eta, *plant.nu, *plant.actual])
        p25_states.append([t, *kinematic_state])
        if i < round(horizon / DT):
            sample = RemusGuidance.step(plant, action, DT)
            controls.append([t, *sample.raw, *sample.applied, *sample.saturated.astype(float)])
            kinematic_state = kinematic_step(kinematic_state, p25_command, DT)
    actual = np.asarray(states)
    p25 = np.asarray(p25_states)
    exact_error = np.linalg.norm(actual[:, 1:4] - prediction.states[:, 1:4], axis=1)
    nominal_error = np.linalg.norm(actual[:, 1:4] - p25[:, 1:4], axis=1)
    numerical_tolerance = 1e-8
    exact_match = bool(np.allclose(actual, prediction.states, atol=numerical_tolerance, rtol=0))
    summary = {
        "task": "open_water_tracking_mismatch_diagnostic", "backend": "remus6dof",
        "obstacle_avoidance_claim": False, "scenario_reachable": True,
        "reference_action": {"depth_m": action.depth_m, "heading_rad": action.heading_rad, "rpm": action.rpm},
        "warmup_s": 20.0, "duration_s": horizon, "physics_dt_s": DT,
        "exact_prediction_matches_actual": exact_match,
        "exact_prediction_numerical_tolerance": numerical_tolerance,
        "six_dof_prediction_position_rmse_m": float(np.sqrt(np.mean(exact_error ** 2))),
        "six_dof_prediction_position_max_error_m": float(exact_error.max()),
        "p25_ideal_rate_position_rmse_m": float(np.sqrt(np.mean(nominal_error ** 2))),
        "p25_ideal_rate_position_final_error_m": float(nominal_error[-1]),
        "p25_ideal_command_u_q_r": p25_command.tolist(),
        "p25_comparison_note": "separate ideal rate-command assumption, not an equivalent depth/heading/RPM command and not REMUS execution",
        "final_depth_target_error_m": float(plant.eta[2] - action.depth_m),
        "final_heading_target_error_deg": float(np.rad2deg(wrap_pi(plant.eta[5] - action.heading_rad))),
        "all_finite": bool(np.isfinite(actual).all() and np.isfinite(prediction.states).all() and np.isfinite(p25).all()),
        "hydrodynamic_parameters_modified": False,
        "passed": exact_match and bool(nominal_error[-1] > 1.0),
        "interpretation": "matching isolated REMUS prediction can replay a fixed reference, while ideal P25 command tracking differs; this is numerical model consistency, not a real-world safety guarantee",
        "wall_clock_s": time.perf_counter() - tic,
    }
    columns = ["time_s", "north_m", "east_m", "depth_m", "roll_rad", "pitch_rad", "yaw_rad", "u_mps", "v_mps", "w_mps", "p_radps", "q_radps", "r_radps", "actual_rudder_rad", "actual_stern_rad", "actual_rpm"]
    np.savetxt(out / "actual_remus.csv", actual, delimiter=",", header=",".join(columns), comments="")
    np.savetxt(out / "predicted_remus.csv", prediction.states, delimiter=",", header=",".join(columns), comments="")
    np.savetxt(out / "ideal_p25.csv", p25, delimiter=",", header="time_s,north_m,east_m,depth_m,u_mps,pitch_rad,yaw_rad", comments="")
    np.savetxt(out / "controls.csv", np.asarray(controls), delimiter=",", header="time_s,raw_rudder_rad,raw_stern_rad,raw_rpm,applied_rudder_rad,applied_stern_rad,applied_rpm,saturated_rudder,saturated_stern,saturated_rpm", comments="")
    np.savetxt(out / "prediction_errors.csv", np.column_stack([actual[:, 0], exact_error, nominal_error]), delimiter=",", header="time_s,six_dof_prediction_position_error_m,ideal_p25_position_error_m", comments="")
    _json(out / "summary.json", summary)
    _plot_diagnostic(actual, prediction.states, p25, exact_error, nominal_error, out)
    return summary


def _plot_diagnostic(actual, predicted, p25, exact_error, nominal_error, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(actual[:, 1], actual[:, 2], actual[:, 3], label="REMUS actual 6-DOF", linewidth=3)
    ax.plot(predicted[:, 1], predicted[:, 2], predicted[:, 3], "--", label="Isolated REMUS prediction", linewidth=1.5)
    ax.plot(p25[:, 1], p25[:, 2], p25[:, 3], ":", label="Ideal P25 [u,q,r] assumption", linewidth=2)
    ax.scatter(*actual[0, 1:4], c="green", label="Common initial position")
    ax.set(xlabel="North (m)", ylabel="East (m)", zlabel="NED depth (m, down +)", title="Open-water reference-step diagnostic")
    ax.invert_zaxis()
    ax.legend(fontsize=7)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(actual[:, 0], exact_error, label="REMUS prediction vs actual")
    ax.plot(actual[:, 0], nominal_error, label="Ideal P25 assumption vs actual")
    ax.set(xlabel="Simulation time (s)", ylabel="3D position error (m)", title="Prediction and execution deviation")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(out / "tracking_diagnostic.png", dpi=140)
    plt.close(fig)
