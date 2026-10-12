"""The risk pilot fixes execution and splits complete independent cases."""
from copy import deepcopy
import json

import numpy as np
import pytest

from auv3d.scenarios import make_stage2_scenario
from auv3d.stage2_config import Stage2Config
from auv3d.stage4_protocol import (
    FAMILIES, METHODS, RISK_LEVELS, SPLITS, common_config,
    config_for_case, manifest, protocol_record, serialize,
)


def test_registration_has_balanced_whole_scene_splits_and_unique_ids():
    rows = manifest()
    assert len(rows) == 48
    assert len({row["case_id"] for row in rows}) == 48
    assert {row["family"] for row in rows} == set(FAMILIES)
    for family in FAMILIES:
        members = [row for row in rows if row["family"] == family]
        assert len(members) == 8
        assert {split: sum(row["split"] == split for row in members) for split in SPLITS} == {
            "train": 1, "tune": 1, "calibration": 4, "test": 2}
        assert len({row["seed"] for row in members}) == 8
    record = protocol_record()
    assert record["development_scene_count"] == 36
    assert record["independent_test_scene_count"] == 12
    assert record["split_counts"] == {"train": 6, "tune": 6, "calibration": 24, "test": 12}


@pytest.mark.parametrize("case", manifest(), ids=lambda row: row["case_id"])
def test_every_case_preserves_common_execution_and_safety(case):
    cfg = config_for_case(case)
    old = Stage2Config()
    assert cfg.physics_dt == old.physics_dt == .02
    assert cfg.control_dt == old.control_dt == 2.
    assert cfg.horizon_s == old.horizon_s == 24.
    assert cfg.candidate_budget == old.candidate_budget == 9
    assert cfg.candidate_scheme == "baseline45"
    assert cfg.shortlist_policy == "diverse"
    assert cfg.emergency_policy == "coast"
    assert cfg.terrain_guidance == "coarse3d_astar"
    assert cfg.obstacle_predictor == "ca"
    assert cfg.safety_margin_m == old.safety_margin_m == 1.
    assert cfg.goal_tolerance_m == old.goal_tolerance_m == 5.
    for name in ("body_radius_m", "depth_limit_m", "nominal_rpm", "rpm_values",
                 "heading_offsets_deg", "depth_offsets_m", "warmup_s"):
        assert getattr(cfg, name) == getattr(old, name)
    assert case["risk_methods"] == list(METHODS)


def test_six_families_vary_real_parameters_instead_of_only_irrelevant_seeds():
    rows = manifest()
    for family in FAMILIES[:-1]:
        members = [row for row in rows if row["family"] == family]
        encoded = {json.dumps(row["config_overrides"], sort_keys=True) for row in members}
        assert len(encoded) == 8
    perlin = [row for row in rows if row["family"] == FAMILIES[-1]]
    assert [row["seed"] for row in perlin] == [7, 11, 13, 17, 19, 23, 29, 31]
    assert all(row["config_overrides"] == {} for row in perlin)


def test_controlled_seed_is_not_misreported_as_geometry_randomization():
    rows = manifest()
    cases = [row for row in rows if row["family"] == "crossing3d"]
    cfg = config_for_case(cases[0])
    a = make_stage2_scenario("crossing3d", cfg)
    changed = deepcopy(cases[0])
    changed["seed"] += 1
    b = make_stage2_scenario("crossing3d", config_for_case(changed))
    np.testing.assert_array_equal(a.obstacles.snapshot().positions, b.obstacles.snapshot().positions)
    a.obstacles.step(40.)
    b.obstacles.step(40.)
    np.testing.assert_array_equal(a.obstacles.snapshot().positions, b.obstacles.snapshot().positions)
    assert cfg.sensor_noise_std_m > 0
    # A second declared speed really does produce another physical trajectory.
    c = make_stage2_scenario("crossing3d", config_for_case(cases[1]))
    c.obstacles.step(40.)
    assert not np.array_equal(a.obstacles.snapshot().positions, c.obstacles.snapshot().positions)


@pytest.mark.parametrize("field,value", [
    ("safety_margin_m", .1), ("goal_tolerance_m", 30.),
    ("physics_dt", .1), ("candidate_budget", 45),
    ("obstacle_predictor", "pplnm"), ("nominal_rpm", 1525.),
    ("prediction_margin_m", 0.), ("max_time_s", 20.),
])
def test_case_cannot_change_protected_or_method_settings(field, value):
    row = manifest()[0]
    row["config_overrides"][field] = value
    with pytest.raises(ValueError, match="method-independent"):
        config_for_case(row)


@pytest.mark.parametrize("change", [
    {"seed": -1}, {"seed": True}, {"seed": 1.5},
    {"split": "random_window"}, {"scenario": "paper_random3d"},
    {"family": "different"}, {"config_overrides": []},
])
def test_malformed_case_is_rejected(change):
    row = manifest()[0]
    row.update(change)
    with pytest.raises(ValueError):
        config_for_case(row)


def test_manifest_is_fresh_and_method_pair_cannot_mutate_another_registration():
    first, second = manifest(), manifest()
    first[0]["config_overrides"]["current_speed"] = 999.
    first[0]["risk_methods"].clear()
    assert second == manifest()
    assert second[0]["config_overrides"]["current_speed"] != 999.
    assert second[0]["risk_methods"] == ["B0", "B1", "B2"]


def test_registration_is_json_finite_reproducible_and_refuses_overwrite(tmp_path):
    target = tmp_path/"protocol"/"EXPERIMENT_CONFIG.json"
    text = serialize(target)
    assert target.read_text(encoding="utf-8") == text
    assert text == serialize()
    record = json.loads(text)
    assert record["risk_levels"] == list(RISK_LEVELS)
    assert record["common_config"] == json.loads(json.dumps(common_config().to_dict()))
    with pytest.raises(FileExistsError):
        serialize(target)
    assert target.read_text(encoding="utf-8") == text


def test_protocol_is_honest_about_missing_obstacles_and_statistical_scope():
    record = protocol_record()
    assert not record["physical_and_safety_parameters_modified"]
    assert not record["future_truth_used_online"]
    assert "correlated" in record["correlation_warning"]
    assert "empirical" in record["correlation_warning"]
    assert "no dynamic obstacles" in record["terrain_negative_control"]
    assert "do not retry seeds" in record["invalid_case_rule"]
