"""New whole-scene registration for risk validation; no outcome-based selection.

Noise replicas share a scene_id and are never additional independent scenes.
Controlled geometry uses one predeclared <=3 m layout translation; original
obstacle counts, radii, motion laws, domain, REMUS and acceptance are unchanged.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .stage2_config import Stage2Config
from .stage4_protocol import common_config as stage4_common_config


PROTOCOL_VERSION = 'stage5_risk_validation_v1'
FAMILIES = ('static3d', 'crossing3d', 'maneuvering3d', 'multi_crossing3d',
            'terrain3d', 'perlin_cruise_constrained3d')
PHASES = ('pilot', 'formal')
SPLITS = ('dev', 'tune', 'calibration', 'test')
METHODS = ('A0', 'A1', 'A2', 'A3', 'A4', 'A5')
PREDICTORS = ('ca', 'cv', 'filtered_cv')
NOISE_LEVELS = (0., .02, .1)
SPEED_BIN_EDGES = (0., .25, float('inf'))
SPEED_BIN_NAMES = ('stationary_observed', 'moving_observed')
PHASE_ALPHA = {'pilot': .2, 'formal': .1}
_PHASE_CALIBRATION_COUNT = {'pilot': 4, 'formal': 12}
_CASE_FIELDS = frozenset(('current_speed', 'current_direction_deg',
    'observation_period_s', 'observation_delay_s', 'sensor_hfov_deg', 'sensor_vfov_deg',
    'sensor_noise_std_m', 'obstacle_speed_scale', 'obstacle_turn_rate_deg_s',
    'terrain_amplitude_m'))


def common_config() -> Stage2Config:
    cfg = replace(stage4_common_config(), label=PROTOCOL_VERSION)
    cfg.validate()
    return cfg


def _noise_value(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError('noise must be a registered finite standard deviation')
    for allowed in NOISE_LEVELS:
        if abs(float(value) - allowed) < 1e-12:
            return allowed
    raise ValueError(f'noise must be one of {NOISE_LEVELS}')


def _base_cases(phase):
    if phase not in PHASES:
        raise ValueError(f'phase must be one of {PHASES}')
    cal_count = _PHASE_CALIBRATION_COUNT[phase]
    split_rows = ['dev', 'tune'] + ['calibration'] * cal_count + ['test'] * 2
    seed_base = 510000 if phase == 'pilot' else 520000
    count = len(split_rows)
    for fi, family in enumerate(FAMILIES):
        for vi, split in enumerate(split_rows):
            seed = seed_base + fi * 1000 + vi
            # Opposite/disjoint pilot/formal lattices, all within original
            # corridor scale. No run result, seed retry or safety search occurs.
            north = -2.8 + 5.6 * (vi + .25 if phase == 'pilot' else vi + .75) / count
            east = -2.7 + 5.4 * ((5 * vi + 3 * fi) % count + .5) / count
            shift = [round(north, 8), round(east, 8), 0.]
            overrides = {}
            if family == 'static3d':
                overrides.update(current_speed=(0., .1, .15, .2)[vi % 4],
                    current_direction_deg=(45., 90., 135., 270.)[(vi // 2) % 4])
            elif family in ('crossing3d', 'multi_crossing3d', 'maneuvering3d'):
                overrides['obstacle_speed_scale'] = (.8, 1., 1.2, 1.5)[vi % 4]
                overrides['observation_period_s'] = (1., 2.)[(vi // 2) % 2]
                overrides['observation_delay_s'] = float((vi // 3) % 2)
                if family == 'maneuvering3d':
                    overrides['obstacle_turn_rate_deg_s'] = (4., 8.)[(vi // 2) % 2]
                if family == 'multi_crossing3d':
                    overrides['sensor_vfov_deg'] = (30., 6.)[(vi // 2) % 2]
            elif family == 'terrain3d':
                overrides['terrain_amplitude_m'] = round(40.5 + 34. * (vi + .5) / count, 8)
                shift = [0., 0., 0.]
            else:
                shift = [0., 0., 0.]
            scene_id = f's5_{phase}_{family}_g{vi:02d}_s{seed}'
            yield dict(scene_id=scene_id, family=family, scenario=family, phase=phase,
                       split=split, seed=seed, variant_index=vi,
                       layout_shift_m=shift, config_overrides=overrides)


def manifest(phase='pilot', split=None, noise_std=None) -> list[dict]:
    """Fresh rows, default three paired noise replicas of each whole geometry.

    pilot: 48 geometries (dev6/tune6/cal24/test12); formal: 96 geometries
    (dev6/tune6/cal72/test12). Noise replicas MUST stay in the same split.
    ``noise_std=.02`` restricts to one noise condition, e.g. the primary formal.
    """
    if split is not None and split not in SPLITS:
        raise ValueError(f'split must be one of {SPLITS}')
    noises = NOISE_LEVELS if noise_std is None else (_noise_value(noise_std),)
    rows = []
    for base in _base_cases(phase):
        if split is not None and base['split'] != split:
            continue
        for noise in noises:
            row = deepcopy(base)
            row['noise_std_m'] = noise
            row['case_id'] = row['scene_id'] + f'_n{round(noise * 1000):03d}'
            row['config_overrides']['sensor_noise_std_m'] = noise
            rows.append(row)
    return rows


def config_for_case(case) -> Stage2Config:
    if not isinstance(case, dict) or case.get('phase') not in PHASES:
        raise ValueError('case requires a registered phase')
    if case.get('split') not in SPLITS or case.get('family') not in FAMILIES or case.get('scenario') != case['family']:
        raise ValueError('case requires a whole-scene split and original family')
    seed = case.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 510000:
        raise ValueError('case requires a new stage5 nonnegative integer seed')
    for key in ('case_id', 'scene_id'):
        value = case.get(key)
        if not isinstance(value, str) or not value or Path(value).name != value or '/' in value or '\\' in value:
            raise ValueError(f'{key} must be a single nonempty directory component')
    overrides = case.get('config_overrides')
    if not isinstance(overrides, dict) or set(overrides) - _CASE_FIELDS:
        raise ValueError('only registered method-independent overrides are permitted')
    noise = _noise_value(case.get('noise_std_m'))
    if overrides.get('sensor_noise_std_m') != noise:
        raise ValueError('case noise and actual sensing configuration disagree')
    shift = np.asarray(case.get('layout_shift_m'), dtype=float)
    if shift.shape != (3,) or not np.isfinite(shift).all() or np.any(np.abs(shift) > 3.) or shift[2] != 0:
        raise ValueError('layout translation must be finite horizontal <=3 m')
    registered = next((base for base in _base_cases(case['phase'])
                       if base['scene_id'] == case['scene_id']), None)
    if registered is None:
        raise ValueError('case identity is outside the frozen phase registration')
    registered['noise_std_m'] = noise
    registered['case_id'] = registered['scene_id'] + f'_n{round(noise * 1000):03d}'
    registered['config_overrides']['sensor_noise_std_m'] = noise
    if any(case.get(key) != value for key, value in registered.items()):
        raise ValueError('case identity, split or parameters differ from frozen registration')
    cfg = replace(common_config(), seed=seed, **deepcopy(overrides))
    cfg.validate()
    return cfg


def make_scene(case, cfg=None):
    """Original scene factory followed by a common predeclared layout shift.

    Scene truth is for sensing, actual collision and offline evaluation only.
    The wrapper does not change motion parameters, methods or select outcomes.
    """
    from .scenarios import make_stage2_scenario
    expected = config_for_case(case)
    cfg = cfg or expected
    # Method-specific risk is injected after the shared full candidate rollout;
    # even the engineering envelope used for fast screening stays identical.
    if cfg.to_dict() != expected.to_dict():
        raise ValueError('scene configuration differs from registered common physical parameters')
    scene = make_stage2_scenario(case['scenario'], cfg)
    shift = np.asarray(case['layout_shift_m'], float)
    obstacles = scene.obstacles
    if len(obstacles.snapshot().ids):
        obstacles.positions = obstacles.positions + shift
        if hasattr(obstacles, 'initial_positions'):
            obstacles.initial_positions = obstacles.initial_positions + shift
    scene.metadata.update(stage5_protocol=PROTOCOL_VERSION, case_id=case['case_id'],
        scene_id=case['scene_id'], phase=case['phase'], split=case['split'],
        common_layout_translation_m=shift.tolist(),
        scene_selection='predeclared whole-scene parameters; no run-result selection or seed retry',
        original_obstacle_counts_radii_motion_laws_preserved=True,
        physical_and_acceptance_parameters_modified=False)
    return scene


make_case_scenario = make_scene


def protocol_record(phase='pilot'):
    cases = manifest(phase, noise_std=.02)
    return dict(protocol_version=PROTOCOL_VERSION, phase=phase,
        unique_geometry_count=len(cases), case_count_with_noise_replicas=len(manifest(phase)),
        split_counts=dict(Counter(row['split'] for row in cases)),
        methods=list(METHODS), predictors=list(PREDICTORS), noise_levels_m=list(NOISE_LEVELS),
        alpha=PHASE_ALPHA[phase], speed_bin_edges_mps=[0., .25, None],
        speed_bin_names=list(SPEED_BIN_NAMES),
        speed_bin_basis='observed last-difference speed; >=.25 m/s is moving; no future speed',
        common_config=common_config().to_dict(),
        independent_unit='complete scene_id, not noise replicas, windows, objects or timesteps',
        calibration='independent predictor/noise libraries from the same new whole-scene pool; empty scenes recorded but excluded from obstacle quantiles; no zero fill for missing labels',
        data_carrier='registered fixed causal observation carrier, not an executed REMUS mission; sensing/ego covariate shift must be reported',
        split_usage={'dev':'interface development only', 'tune':'A1 selection and declared pilot choices',
                     'calibration':'frozen offline labels only', 'test':'no threshold, predictor or calibration fitting'},
        invalid_case_rule='record original ID and factory error; no seed retry or geometry replacement',
        physical_and_safety_parameters_modified=False, future_truth_used_online=False,
        stage4_data_preserved=True, historical_stage4_library='separate CA sensitivity only; never mixed into primary matched-calibration comparison',
        formal_primary='CA, noise .02, alpha .1, paired twelve test geometries and all six methods; other settings are separately declared subsets',
        cases=cases)


def serialize(path=None, phase='pilot'):
    value = json.dumps(protocol_record(phase), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(value)
    return value
