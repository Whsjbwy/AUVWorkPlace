"""Pre-registered, method-independent cases for the stage 4 risk pilot.

This file describes an experiment, not a new obstacle motion model.  Controlled
stage 2 scenes ignore their random seed when constructing geometry.  Their
cases therefore vary actual flow, sensing, speed, or turn parameters as well as
the measurement seed.  Perlin seeds genuinely change the original seabed.

The complete scene is the split unit.  Windows from a scene may never move to
another split.  The small stratified pilot is not an exchangeability proof or a
certificate of closed-loop collision probability.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Mapping

from .stage2_config import Stage2Config


PROTOCOL_VERSION = "stage4_risk_pilot_v1"
FAMILIES = (
    "static3d", "crossing3d", "maneuvering3d", "multi_crossing3d",
    "terrain3d", "perlin_cruise_constrained3d",
)
SPLITS = ("train", "tune", "calibration", "test")
METHODS = ("B0", "B1", "B2")
RISK_LEVELS = (0.10, 0.20, 0.30)

# Split assignments are fixed before results exist, balanced within each family,
# and deliberately do not put all highest speed/current variants in one split.
_SPLIT_ROWS = (
    ("calibration", "test", "train", "calibration", "tune", "calibration", "test", "calibration"),
    ("train", "calibration", "test", "calibration", "calibration", "tune", "calibration", "test"),
    ("calibration", "train", "calibration", "test", "test", "calibration", "tune", "calibration"),
    ("test", "calibration", "tune", "calibration", "calibration", "test", "calibration", "train"),
    ("calibration", "tune", "calibration", "train", "test", "calibration", "test", "calibration"),
    ("tune", "calibration", "calibration", "test", "calibration", "train", "calibration", "test"),
)

# Only these method-independent experimental settings can vary across cases.
# All physical coefficients and safety/arrival definitions remain outside this
# list.  A risk method is selected separately by the stage 4 runner.
_CASE_FIELDS = frozenset((
    "current_speed", "current_direction_deg", "sensor_noise_std_m",
    "sensor_hfov_deg", "sensor_vfov_deg", "sensor_range_m",
    "observation_period_s", "observation_delay_s", "obstacle_speed_scale",
    "obstacle_turn_rate_deg_s", "terrain_amplitude_m",
))


def common_config() -> Stage2Config:
    """Original REMUS and reference budget, shared by B0, B1 and B2."""
    cfg = replace(
        Stage2Config(),
        physics_dt=0.02, control_dt=2.0, horizon_s=24.0,
        prediction_output_dt=0.2, candidate_budget=9,
        candidate_scheme="baseline45", shortlist_policy="diverse",
        emergency_policy="coast", memory_policy="hard_expiry",
        track_max_age_s=4.0, terrain_guidance="coarse3d_astar",
        obstacle_predictor="ca", prediction_history_dt_s=1.0,
        safety_margin_m=1.0, goal_tolerance_m=5.0, max_time_s=200.0,
        label=PROTOCOL_VERSION,
    )
    cfg.validate()
    return cfg


def _variants() -> tuple[tuple[dict, ...], ...]:
    """Fixed values; no planner result, acceptance test, or seed retry here."""
    static = tuple(
        dict(current_speed=current, current_direction_deg=direction,
             sensor_noise_std_m=0.02, observation_delay_s=delay)
        for current, direction, delay in (
            (0.0, 90.0, 0.0), (0.1, 45.0, 0.0), (0.1, 90.0, 1.0),
            (0.1, 135.0, 0.0), (0.15, 90.0, 1.0), (0.15, 270.0, 0.0),
            (0.2, 45.0, 0.0), (0.2, 135.0, 1.0),
        )
    )
    crossing = tuple(
        dict(obstacle_speed_scale=speed, observation_period_s=period,
             observation_delay_s=delay, sensor_noise_std_m=0.02)
        for speed, period, delay in (
            (1.0, 1.0, 0.0), (0.75, 1.0, 1.0), (1.25, 1.0, 0.0),
            (1.5, 1.0, 1.0), (1.0, 2.0, 0.0), (0.75, 2.0, 1.0),
            (1.25, 2.0, 1.0), (1.5, 2.0, 0.0),
        )
    )
    maneuvering = tuple(
        dict(obstacle_speed_scale=speed, obstacle_turn_rate_deg_s=turn,
             sensor_noise_std_m=0.02)
        for speed, turn in (
            (0.8, 4.0), (1.0, 4.0), (1.2, 4.0), (1.4, 4.0),
            (0.8, 8.0), (1.0, 8.0), (1.2, 8.0), (1.4, 8.0),
        )
    )
    multiple = tuple(
        dict(obstacle_speed_scale=speed, sensor_vfov_deg=vfov,
             observation_period_s=period, observation_delay_s=delay,
             sensor_noise_std_m=0.02)
        for speed, vfov, period, delay in (
            (1.0, 30.0, 1.0, 0.0), (0.8, 6.0, 1.0, 0.0),
            (1.2, 30.0, 2.0, 1.0), (1.5, 30.0, 1.0, 0.0),
            (1.0, 6.0, 2.0, 0.0), (0.8, 30.0, 2.0, 1.0),
            (1.2, 6.0, 1.0, 1.0), (1.5, 6.0, 2.0, 1.0),
        )
    )
    # No obstacle forecasts exist in these two families.  They are a negative
    # control: risk-only changes should not change their guidance or execution.
    ridge = tuple(dict(terrain_amplitude_m=float(amplitude))
                  for amplitude in (40, 45, 50, 55, 60, 65, 70, 75))
    perlin = tuple({} for _ in range(8))
    return static, crossing, maneuvering, multiple, ridge, perlin


def manifest() -> list[dict]:
    """Return 48 fresh JSON-compatible rows: 36 development and 12 test cases.

    ``config_overrides`` never contains method or risk settings.  The same row
    is supplied to every method.  Returning new dictionaries prevents one
    caller from modifying the registration used by another method.
    """
    rows = []
    perlin_seeds = (7, 11, 13, 17, 19, 23, 29, 31)
    for family_index, (family, variants, split_row) in enumerate(
            zip(FAMILIES, _variants(), _SPLIT_ROWS, strict=True)):
        for variant_index, (overrides, split) in enumerate(zip(variants, split_row, strict=True)):
            seed = (perlin_seeds[variant_index] if family == "perlin_cruise_constrained3d"
                    else 410000 + 100*family_index + variant_index)
            if family == "static3d":
                diversity = "same original static obstacle; actual horizontal current and noisy delivered observations vary"
            elif family in ("crossing3d", "maneuvering3d", "multi_crossing3d"):
                diversity = "original positions retained; predeclared speed/turn/sensing settings and independent noisy observations vary"
            elif family == "terrain3d":
                diversity = "original analytic ridge formula and endpoints; predeclared ridge amplitude varies"
            else:
                diversity = "original Perlin map seed varies; unchanged original-model neutral-cruise initialization admission, without seed retry"
            rows.append({
                "case_id": f"{family}_v{variant_index:02d}_s{seed}",
                "family": family, "scenario": family,
                "variant_index": variant_index, "split": split, "seed": seed,
                "config_overrides": deepcopy(overrides),
                "diversity_notes": diversity,
                "risk_methods": list(METHODS),
            })
    return rows


def config_for_case(case: Mapping) -> Stage2Config:
    """Build the common execution configuration, rejecting safety mutations."""
    if not isinstance(case, Mapping):
        raise ValueError("case must be a registered scene mapping")
    if case.get("scenario") not in FAMILIES or case.get("family") != case.get("scenario"):
        raise ValueError("case must name one of the six original scene families")
    if case.get("split") not in SPLITS:
        raise ValueError(f"case split must be one of {SPLITS}")
    seed = case.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("case seed must be a nonnegative integer")
    overrides = case.get("config_overrides", {})
    if not isinstance(overrides, Mapping) or set(overrides) - _CASE_FIELDS:
        raise ValueError("case overrides may change only the registered method-independent experimental settings")
    cfg = replace(common_config(), seed=seed, **dict(overrides))
    cfg.validate()
    return cfg


def protocol_record() -> dict:
    """Full pre-registration; coverage statements remain empirical pilot scope."""
    cases = manifest()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "registered_scene_count": len(cases),
        "development_scene_count": sum(case["split"] != "test" for case in cases),
        "independent_test_scene_count": sum(case["split"] == "test" for case in cases),
        "split_counts": dict(Counter(case["split"] for case in cases)),
        "methods": {"B0": "CA + retained fixed prediction margin",
                    "B1": "same CA + P25-style independently calibrated error balls",
                    "B2": "same CA + ECP-inspired candidate-dependent risk calibration"},
        "risk_levels": list(RISK_LEVELS),
        "independent_test_risk_level": 0.10,
        "common_config": common_config().to_dict(),
        "cases": cases,
        "split_unit": "whole registered scene; no overlapping history/future windows may cross a split",
        "split_usage": {
            "train": "feature and data-contract development only; CA has no fitted weights",
            "tune": "predeclared safety-efficiency comparison and any method choices; never independent test tuning",
            "calibration": "common B0 delivered histories and offline error labels; used by both B1 and B2",
            "test": "paired final B0/B1/B2 closed loops; no parameter or calibration updates from test future truth",
        },
        "calibration_rule": "build new current-CA 24-second calibration from common B0 delivered tracks; create future error labels only offline after acquisition; preserve whole scene blocks and align stored error trajectories by observed ego yaw",
        "risk_envelope_rule": "all methods keep the original 0.75 m plus age/memory prediction margin; B1 and B2 only add their independently calibrated risk term, without changing the physical sphere or 1 m safety threshold",
        "calibration_block_rule": "one score per complete calibration scene, taking maxima over its matured anchors and forecast intervals; 24 registered scene blocks, not a claim of 24 independent dynamic-obstacle blocks",
        "risk_time_grid": "24 s forecast, 0.2 s calibration grid; final candidate and collision verification retain every original 0.02 s physical step",
        "test_threshold_rule": "test alpha is fixed at 0.1 before evaluation, regardless of tune results; tune curves use all registered alpha values without selecting a favorable test threshold",
        "correlation_warning": "many windows from one scene are correlated; window-level empirical coverage is not a finite-sample whole-task guarantee",
        "online_information": "only delivered causal observations; frozen offline ECP adaptation with no ACP online update in this pilot; any later update would require already matured and delivered labels",
        "truth_information": "offline calibration/evaluation and actual collision monitor only; never online risk scoring or candidate ranking",
        "occlusion_scope": "body-fixed narrow-FOV multi-obstacle cases can cause temporary loss; report measured loss counts, do not assert an unmeasured fixed dropout duration",
        "terrain_negative_control": "both terrain families contain no dynamic obstacles; risk-only methods must retain the same guidance and plant response",
        "invalid_case_rule": "record a factory/admission error against the original ID; do not retry seeds, change positions, or silently count a replacement scene",
        "timing_scope": "same 9 full candidate rollouts plus original coast fallback; planning wall time is measured and not silently converted to physical delay",
        "physical_and_safety_parameters_modified": False,
        "future_truth_used_online": False,
    }


def serialize(path: str | Path | None = None) -> str:
    """Serialize pre-registration, optionally creating a new file exclusively."""
    value = json.dumps(protocol_record(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
    return value
