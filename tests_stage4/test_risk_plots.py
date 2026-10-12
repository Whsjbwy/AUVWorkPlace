"""Fair pairing and honest risk-trace scopes in offline plot helpers."""
from copy import deepcopy

import numpy as np
import pytest

from auv3d.stage4_plots import _risk_trace, _validate_pair, _number, _metric_group


def _case():
    return {'config': {'seed': 7, 'safety_margin_m': 1., 'physics_dt': .02,
                       'candidate_budget': 9, 'label': 'B0'},
            'summary': {'scenario': 'crossing3d', 'seed': 7, 'goal_ned_m': [210., 100., 35.]},
            'terrain': {'xs': np.array([0., 260.]), 'ys': np.array([0., 200.]),
                        'depths': np.full((2, 2), 90.)}}


def test_paired_common_parameters_and_terrain_must_match():
    first, second = _case(), deepcopy(_case())
    second['config']['label'] = 'B1 provenance only'
    _validate_pair({'B0': first, 'B1': second})
    second['config']['safety_margin_m'] = .5
    with pytest.raises(ValueError, match='configuration fields'):
        _validate_pair({'B0': first, 'B1': second})
    second = deepcopy(first)
    second['terrain']['depths'][0, 0] += 1.
    with pytest.raises(ValueError, match='terrain arrays'):
        _validate_pair({'B0': first, 'B1': second})


def test_candidate_quantile_trace_excludes_emergency_and_keeps_unknown():
    plans = [{'time_s': 0., 'risk_records': [
        {'action_kind': 'candidate', 'quantile_m': 1.},
        {'action_kind': 'candidate', 'quantile_m': 3.},
        {'action_kind': 'emergency_coast', 'quantile_m': 99.}]},
        {'time_s': 2., 'risk_records': []}]
    times, lower, upper = _risk_trace(plans, 'B2')
    assert np.array_equal(times, [0., 2.])
    assert lower[0] == 1. and upper[0] == 3.
    assert np.isnan(lower[1]) and np.isnan(upper[1])
    assert _risk_trace([{'time_s': 2.}], 'B0')[1][0] == 0.


def test_unknown_metric_is_not_a_success_or_zero_collision():
    assert _number('') is None
    assert _number('None') is None
    assert _number('nan') is None
    assert _number('False') == 0.
    assert _number('True') == 1.
    rows = [{'navigation_success': 'True', 'collision': 'False'},
            {'navigation_success': 'False', 'collision': 'True'},
            {'navigation_success': '', 'collision': ''}]
    assert _metric_group('navigation_success', rows) == (1 / 3, 2, 'confirmed_arrivals_over_all_registered_results')
    assert _metric_group('collision', rows) == (.5, 2, 'mean_known_episode_indicator')
