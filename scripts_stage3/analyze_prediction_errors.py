"""Paired offline CA/PPLNM error evaluation from delivered REMUS histories.

This program has no planner feedback or vehicle rollout. Same-seed future
obstacle truth is generated only after execution and checked against every
saved truth snapshot. Shared anchors/history/timestamps prevent comparisons of
different observed objects or different prediction origins.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from numbers import Integral
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auv3d.sensors import Track, validate_track
from auv3d.stage2_config import Stage2Config
from auv3d.stage2_failure_analysis import _truth_at_times
from auv3d.stage3_forecasting import ForecastContext, make_obstacle_forecaster


REQUIRED = ("summary.json", "config.json", "planning_diagnostics.json",
            "observations.json", "obstacles.json")
HORIZONS_S = (3.0, 6.0, 9.0, 12.0, 18.0, 24.0)
SCOPE = ("Offline paired prediction errors, not navigation success or risk calibration. "
         "Every predictor receives exactly the same already delivered measured history "
         "at the same planning time and absolute forecast timestamps. Ground truth is "
         "read/generated only after execution; no feedback, fitting, threshold tuning "
         "or new vehicle rollout occurs. A 9-second trained range is measured from "
         "the last observation, not from the later planning timestamp.")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tick(stamp, dt):
    if not isinstance(stamp, (int, float)) or not np.isfinite(stamp):
        raise ValueError("saved timestamp must be finite")
    tick = round(stamp / dt)
    if tick < 0 or not np.isclose(tick*dt, stamp, atol=1e-8, rtol=0):
        raise ValueError("saved timestamp must align with the original physics tick")
    return tick


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(_json_safe(value), stream, indent=2, ensure_ascii=False, allow_nan=False)


def _write_csv(path, rows, empty_columns):
    columns = list(dict.fromkeys(key for row in rows for key in row)) or list(empty_columns)
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _track_from_saved(row, now):
    object_id = row.get("object_id")
    if not isinstance(object_id, Integral) or isinstance(object_id, bool):
        raise ValueError("saved track object_id must be an integer")
    radius = row.get("radius", row.get("radius_m"))
    times = row.get("times_s", row.get("times"))
    try:
        track = Track(int(object_id), float(radius), list(times),
                      [np.asarray(point, dtype=float) for point in row["positions"]])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("saved causal track is malformed") from error
    validate_track(track, now)
    return track


def _validate_delivered_history(track, now, observation_packets, dt):
    """Require every saved measurement to have actually been delivered by now."""
    delivered = {}
    for packet in observation_packets:
        measured = float(packet["measurement_time_s"])
        measured_tick = _tick(measured, dt)
        delivery = packet.get("delivered_time_s")
        if delivery is None:
            continue
        _tick(delivery, dt)
        if delivery < measured - 1e-9:
            raise ValueError("an observation was delivered before its acquisition")
        if delivery > now + 1e-9:
            continue
        for observed in packet["objects"]:
            object_id = observed["object_id"]
            if not isinstance(object_id, Integral) or isinstance(object_id, bool):
                raise ValueError("delivered object_id must be an integer")
            key = (int(object_id), measured_tick)
            if key in delivered:
                raise ValueError("duplicate delivered observation in saved source")
            position = np.asarray(observed["position"], dtype=float)
            radius = observed.get("radius", observed.get("radius_m"))
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("invalid delivered observation position")
            if not isinstance(radius, (int, float)) or not np.isfinite(radius) or radius <= 0:
                raise ValueError("invalid delivered observation radius")
            delivered[key] = (position, float(radius))
    for i, (stamp, position) in enumerate(zip(track.times, track.positions)):
        key = (track.object_id, _tick(stamp, dt))
        if key not in delivered:
            raise ValueError("saved history contains an observation not delivered at its planning anchor")
        measured_position, radius = delivered[key]
        if not np.array_equal(measured_position, position):
            raise ValueError("saved history position differs from its delivered observation")
        if i == len(track.times)-1 and radius != track.radius_m:
            raise ValueError("saved track radius differs from latest delivered observation")


def _forecasters(cfg, checkpoint=None, calibration=None, extension_modes=("cv_after9", "ca_after9")):
    result = {"ca_dt3": make_obstacle_forecaster(replace(
        cfg, obstacle_predictor="ca", prediction_history_dt_s=3.0))}
    for mode in extension_modes:
        if mode not in ("cv_after9", "ca_after9"):
            raise ValueError("offline paired comparison requires explicit cv_after9 or ca_after9")
        options = dict(obstacle_predictor="pplnm", prediction_history_dt_s=3.0,
                       pplnm_extension_mode=mode, pplnm_threads=2)
        if checkpoint is not None:
            options["pplnm_checkpoint"] = str(checkpoint)
        if calibration is not None:
            options["pplnm_calibration"] = str(calibration)
        result["pplnm_" + mode] = make_obstacle_forecaster(replace(cfg, **options))
    return result


def paired_case(source, *, forecasters=None, horizons=HORIZONS_S,
                checkpoint=None, calibration=None,
                extension_modes=("cv_after9", "ca_after9")):
    """Read/validate one completed episode, then evaluate identical anchors."""
    source = Path(source)
    if not all((source/name).is_file() for name in REQUIRED):
        raise FileNotFoundError(f"incomplete saved REMUS episode: {source}")
    cfg = Stage2Config.load(source/"config.json")
    saved_summary = read_json(source/"summary.json")
    if saved_summary.get("backend") != "remus6dof":
        raise ValueError("paired REMUS analysis requires an actual remus6dof source")
    scenario = saved_summary["scenario"]
    if saved_summary.get("seed") != cfg.seed:
        raise ValueError("saved scenario seed differs from source configuration")
    horizons = np.asarray(horizons, dtype=float)
    if (horizons.ndim != 1 or not len(horizons) or not np.isfinite(horizons).all()
            or np.any(horizons <= 0) or np.any(np.diff(horizons) <= 0)):
        raise ValueError("comparison horizons must be finite, positive and increasing")
    for horizon in horizons:
        _tick(float(horizon), cfg.physics_dt)
    observations = read_json(source/"observations.json")
    details = read_json(source/"planning_diagnostics.json")
    recorded = read_json(source/"obstacles.json")
    if not recorded:
        raise ValueError("offline regenerated truth requires recorded snapshots for verification")
    anchors, requested_ticks = [], set()
    previous_time = -1.0
    for detail in details:
        now = float(detail["time_s"])
        _tick(now, cfg.physics_dt)
        if now <= previous_time:
            raise ValueError("planning anchors must be strictly increasing")
        previous_time = now
        tracks = [_track_from_saved(row, now) for row in detail.get("tracks", detail.get("causal_tracks", []))]
        if len({track.object_id for track in tracks}) != len(tracks):
            raise ValueError("duplicate object ID within one planning anchor")
        for track in tracks:
            _validate_delivered_history(track, now, observations, cfg.physics_dt)
        times = now + horizons
        requested_ticks.update(_tick(float(stamp), cfg.physics_dt) for stamp in times)
        uncertainty = {int(key): value for key, value in detail.get("memory_uncertainty_by_id", {}).items()}
        anchors.append((now, tracks, times, uncertainty))
    truth, verification = _truth_at_times(cfg, scenario, requested_ticks, recorded)
    variants = forecasters or _forecasters(cfg, checkpoint, calibration, extension_modes)
    if not variants:
        raise ValueError("at least one predictor is required for paired analysis")
    contexts = {name: ForecastContext(cfg, predictor) for name, predictor in variants.items()}
    rows, anchor_records = [], []
    duration = float(saved_summary["duration_s"])
    for anchor_index, (now, tracks, times, uncertainty) in enumerate(anchors):
        for context in contexts.values():
            context.begin(uncertainty)
        for track in tracks:
            raw = dict(anchor_index=anchor_index, planning_time_s=now,
                       object_id=track.object_id, latest_measurement_time_s=track.times[-1],
                       measured_times_s=track.times, measured_positions=np.asarray(track.positions).tolist(),
                       physical_obstacle_radius_m=track.radius_m,
                       memory_extra_uncertainty_m=uncertainty.get(track.object_id, 0), forecasts={})
            for variant, context in contexts.items():
                result = context.forecast(track, times, now)
                metadata = result.metadata
                trained_horizon = metadata.get("trained_horizon_s")
                if trained_horizon is not None and (not np.isfinite(trained_horizon) or trained_horizon <= 0):
                    raise ValueError("trained predictor horizon must be finite and positive")
                trained_mask = None
                if trained_horizon is not None:
                    trained_mask = times - track.times[-1] <= trained_horizon + 1e-8
                    claimed = metadata.get("trained_valid_mask")
                    if claimed is not None and not np.array_equal(np.asarray(claimed), trained_mask):
                        raise ValueError("trained validity mask does not match the last measurement time anchor")
                raw["forecasts"][variant] = dict(absolute_times_s=times.tolist(),
                    positions=result.positions.tolist(), enclosing_radii_m=result.enclosing_radii_m.tolist(),
                    metadata=_json_safe(metadata))
                for index, (horizon, target_time) in enumerate(zip(horizons, times)):
                    snapshot = truth[_tick(float(target_time), cfg.physics_dt)]
                    matches = np.flatnonzero(snapshot.ids == track.object_id)
                    if len(matches) != 1:
                        raise ValueError("saved tracked object does not have exactly one offline truth match")
                    truth_position = snapshot.positions[int(matches[0])]
                    predicted = result.positions[index]
                    error = float(np.linalg.norm(predicted - truth_position))
                    within = None if trained_mask is None else bool(trained_mask[index])
                    scope = ("ca_untrained_reference" if within is None else
                             "within_trained_range_from_measurement" if within else "explicit_post_training_extension")
                    common_scope = ("within_pplnm_9s_from_measurement"
                        if target_time-track.times[-1] <= 9.+1e-9
                        else "post_pplnm_9s_from_measurement")
                    calibration = metadata.get("p25_calibration_radii_m")
                    calibration_radius = None if calibration is None else calibration[index]
                    rows.append(dict(case_id=source.name, scenario=scenario, scenario_seed=cfg.seed,
                        source_navigation_status=saved_summary["status"],
                        memory_policy=cfg.memory_policy, candidate_budget=cfg.candidate_budget,
                        terrain_guidance=cfg.terrain_guidance, variant=variant,
                        predictor=metadata.get("predictor", context.predictor.name),
                        extension_mode=metadata.get("extension_mode", "not_applicable"),
                        anchor_index=anchor_index, planning_time_s=now, object_id=track.object_id,
                        latest_measurement_time_s=float(track.times[-1]), observation_age_s=now-track.times[-1],
                        planning_horizon_s=float(horizon), from_measurement_horizon_s=float(target_time-track.times[-1]),
                        target_absolute_time_s=float(target_time), trained_horizon_s=trained_horizon,
                        within_trained_horizon=within, training_scope=scope, model_scope=scope,
                        common_pplnm_support_scope=common_scope,
                        target_after_episode_end=bool(target_time > duration+1e-9),
                        predicted_north_m=float(predicted[0]), predicted_east_m=float(predicted[1]),
                        predicted_depth_m=float(predicted[2]), truth_north_m=float(truth_position[0]),
                        truth_east_m=float(truth_position[1]), truth_depth_m=float(truth_position[2]),
                        position_error_m=error, enclosing_radius_m=float(result.enclosing_radii_m[index]),
                        p25_calibration_radius_m=calibration_radius,
                        calibrated_radii_applied=bool(metadata.get("calibrated_radii_applied", False))))
            anchor_records.append(raw)
    summary = dict(case_id=source.name, scenario=scenario, scenario_seed=cfg.seed,
                   source_directory=str(source), source_navigation_status=saved_summary["status"],
                   source_obstacle_predictor=cfg.obstacle_predictor,
                   planning_anchor_count=len(anchors), object_anchor_count=len(anchor_records),
                   paired_row_count=len(rows), predictors=list(variants), horizons_s=horizons.tolist(),
                   ca_history_dt_s=3.0, trained_range_time_anchor="last delivered observation acquisition time",
                   new_vehicle_rollouts=0, feedback_to_planner=False, parameter_fitting=False,
                   future_truth_online=False, scope=SCOPE, **verification)
    return rows, anchor_records, summary


def error_statistics(rows):
    """Use identical nine-second support subsets for CA and learned variants.

    The CA reference has no trained horizon of its own; giving it the same
    measurement-age subset avoids comparing all CA samples to only fresh
    PPLNM samples. Original per-model scope remains an explicit trace field.
    """
    grouped = {}
    for row in rows:
        key = (row["case_id"], row["variant"], row["planning_horizon_s"], row["common_pplnm_support_scope"])
        grouped.setdefault(key, []).append(row)
    statistics = []
    for key, group in sorted(grouped.items()):
        errors = np.array([row["position_error_m"] for row in group])
        if not np.isfinite(errors).all():
            raise ValueError("prediction errors must be finite")
        row = group[0]
        statistics.append(dict(case_id=key[0], scenario=row["scenario"], scenario_seed=row["scenario_seed"],
            memory_policy=row["memory_policy"], candidate_budget=row["candidate_budget"],
            terrain_guidance=row["terrain_guidance"], variant=key[1], planning_horizon_s=key[2],
            common_pplnm_support_scope=key[3], training_scope=row["training_scope"],
            model_scope=row["model_scope"], sample_count=len(group),
            prediction_RMSE_m=float(np.sqrt(np.mean(errors**2))),
            prediction_max_error_m=float(errors.max()), prediction_P95_error_m=float(np.percentile(errors, 95)),
            minimum_from_measurement_horizon_s=min(item["from_measurement_horizon_s"] for item in group),
            maximum_from_measurement_horizon_s=max(item["from_measurement_horizon_s"] for item in group),
            targets_after_episode_end_count=sum(item["target_after_episode_end"] for item in group)))
    return statistics


def analyze_sources(sources, out, *, forecasters=None, horizons=HORIZONS_S,
                    checkpoint=None, calibration=None,
                    extension_modes=("cv_after9", "ca_after9"), plot=True):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"choose a new output directory: {out}")
    sources = [Path(source) for source in sources]
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("source case directory names must be distinct; do not merge incomparable runs")
    paired, anchors, summaries = [], [], []
    for source in sources:
        rows, records, summary = paired_case(source, forecasters=forecasters, horizons=horizons,
            checkpoint=checkpoint, calibration=calibration, extension_modes=extension_modes)
        paired.extend(rows)
        anchors.extend(dict(case_id=source.name, **record) for record in records)
        summaries.append(summary)
    statistics = error_statistics(paired)
    common_anchor_sets = {}
    for row in paired:
        key = (row["case_id"], row["planning_horizon_s"], row["common_pplnm_support_scope"])
        common_anchor_sets.setdefault(key, {}).setdefault(row["variant"], set()).add(
            (row["anchor_index"], row["object_id"]))
    for case_domain, by_variant in common_anchor_sets.items():
        sample_sets = list(by_variant.values())
        if any(sample_set != sample_sets[0] for sample_set in sample_sets[1:]):
            raise ValueError(f"predictor comparison has different observed anchors: {case_domain}")
    out.mkdir(parents=True, exist_ok=False)
    _write_csv(out/"paired_prediction_errors.csv", paired,
               ["case_id", "scenario", "scenario_seed", "variant", "planning_horizon_s", "position_error_m"])
    _write_csv(out/"prediction_RMSE_by_horizon.csv", statistics,
               ["case_id", "scenario", "scenario_seed", "variant", "planning_horizon_s", "prediction_RMSE_m"])
    _write_json(out/"prediction_anchors.json", anchors)
    result = dict(scope=SCOPE, cases=summaries, paired_row_count=len(paired),
                  statistics_row_count=len(statistics), model_parameters_adjusted=False,
                  risk_margins_adjusted=False, calibration_refitted=False,
                  ground_truth_used_for_prediction=False,
                  statistics_grouping=["case_id", "variant", "planning_horizon_s", "common_pplnm_support_scope"],
                  common_support_anchor_sets_equal=True)
    _write_json(out/"summary.json", result)
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for case in sources:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for variant in sorted({row["variant"] for row in paired if row["case_id"] == case.name}):
                # Combine samples only at the same explicit case/variant/horizon.
                values = []
                for horizon in horizons:
                    errors = [row["position_error_m"] for row in paired
                              if row["case_id"] == case.name and row["variant"] == variant
                              and row["planning_horizon_s"] == horizon]
                    if errors:
                        values.append((horizon, np.sqrt(np.mean(np.square(errors)))))
                if values:
                    ax.plot(*zip(*values), marker="o", label=variant)
            ax.axvline(9, linestyle=":", color="grey", label="9 s from planning; stale histories end trained support earlier")
            ax.set(xlabel="Horizon from common planning anchor (s)", ylabel="Paired position RMSE (m)",
                   title=case.name + "\nOffline errors; trained/extension ranges separated in CSV")
            ax.grid(alpha=.25)
            if ax.lines:
                ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out/(case.name+"_prediction_rmse.png"), dpi=150)
            plt.close(fig)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", required=True, help="completed REMUS episode directories")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--calibration")
    parser.add_argument("--extension-modes", nargs="+", choices=["cv_after9", "ca_after9"],
                        default=["cv_after9", "ca_after9"])
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    result = analyze_sources(args.source, args.out, checkpoint=args.checkpoint,
                             calibration=args.calibration, extension_modes=args.extension_modes,
                             plot=not args.no_plot)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
