from copy import deepcopy
import json

import numpy as np
import pytest

from auv3d.stage5_calibration import QUERY_TIMES_S, generate_evaluation_samples
from auv3d.stage5_forecasting import forecaster_contract
from auv3d.stage5_protocol import PREDICTORS, config_for_case, manifest
from scripts_stage5.prediction_noise import (
    aggregate_coverage, aggregate_rmse, evaluate_case, forecast_source,
    load_calibration, load_registry, run_diagnostics,
)


def _case(noise=.02):
    return next(r for r in manifest(split='dev', noise_std=noise) if r['family'] == 'static3d')


def _library(path, predictor='ca', noise=.02, mutation=None):
    cases = manifest(split='calibration', noise_std=noise)[:8]
    shape = (8, 1, 2, len(QUERY_TIMES_S), 3)
    pred, label = np.zeros(shape), np.zeros(shape)
    for si in range(8):
        label[si, :, :, :, 0] = si+1
    mask = np.ones(shape[:3], bool)
    speeds = np.zeros(mask.shape)
    speeds[:, :, 1] = 1.
    ids = [r['case_id'] for r in cases]
    scene_ids = [r['scene_id'] for r in cases]
    metadata = dict(schema='stage5-risk-v1', split='calibration', phase='pilot',
        predictor=predictor, predictor_contract=forecaster_contract(predictor, config_for_case(_case(noise))),
        noise_std_m=noise, case_ids=ids, speed_bin_edges_mps=[0., .25, 'inf'],
        scenes=[dict(case_id=r['case_id'], scene_id=r['scene_id'], phase=r['phase'],
                     noise_std_m=noise, split=r['split']) for r in cases])
    if mutation:
        mutation(metadata)
    np.savez_compressed(path, pred_local=pred, label_local=label, mask=mask,
        observed_speeds_mps=speeds, times_s=QUERY_TIMES_S,
        case_ids=np.asarray(ids), scene_ids=np.asarray(scene_ids), empty_scene_mask=np.zeros(8, bool),
        metadata_json=np.asarray(json.dumps(metadata)))
    return path


def test_matched_calibration_quantiles_and_unseen_roles(tmp_path):
    path = _library(tmp_path/'cal.npz')
    result = load_calibration(path, 'ca', _case(), .2)
    assert result['joint_q'] == 8.
    assert result['joint_scene_count'] == 8
    assert result['pointwise']['scene_counts'].tolist() == [8, 8]
    assert (result['pointwise']['quantiles_m'] == 8.).all()
    with pytest.raises(ValueError, match='predictor'):
        load_calibration(path, 'cv', _case(), .2)
    with pytest.raises(ValueError, match='noise_std_m'):
        load_calibration(path, 'ca', _case(.1), .2)


@pytest.mark.parametrize('mutator', [
    lambda m: m.update(split='test'),
    lambda m: m['scenes'][0].update(split='test'),
    lambda m: m['scenes'][0].update(scene_id='wrong'),
])
def test_source_test_role_or_identity_cannot_hide_under_calibration_header(tmp_path, mutator):
    path = _library(tmp_path/'bad.npz', mutation=mutator)
    with pytest.raises(ValueError):
        load_calibration(path, 'ca', _case(), .2)


def test_predictors_never_receive_offline_future_labels():
    case = _case(0.)
    sample = generate_evaluation_samples(case)
    original = forecast_source(case, sample)
    changed = deepcopy(sample)
    for window in changed['windows']:
        window['label_world'] += 1000000.
    diagnostic = forecast_source(case, changed)
    np.testing.assert_array_equal(original['pred_world'], diagnostic['pred_world'])
    assert not np.array_equal(original['label_world'], diagnostic['label_world'])
    for record in original['histories']:
        assert max(record['acquisition_times_s']) <= record['anchor_s']


def test_rmse_aggregation_weights_label_count_not_case_rmse():
    common = dict(phase='pilot', split='dev', predictor='ca', noise_std_m=.02, horizon_s=24.)
    rows = [dict(common, position_label_count=1, squared_euclidean_error_sum_m2=9., prediction_RMSE_m=3.),
            dict(common, position_label_count=9, squared_euclidean_error_sum_m2=144., prediction_RMSE_m=4.)]
    result = aggregate_rmse(rows)[0]
    assert result['position_label_count'] == 10
    assert result['prediction_RMSE_m'] == pytest.approx(np.sqrt(153/10))
    assert result['prediction_RMSE_m'] != 3.5


def test_empty_observations_have_unknown_coverage_not_success(tmp_path):
    case = _case()
    calibrations = {p: load_calibration(_library(tmp_path/f'{p}.npz', p), p, case, .2) for p in PREDICTORS}
    forecasts = dict(mask=np.zeros((3, 1), bool), observed_speeds_mps=np.zeros((3, 1)),
        label_world=np.zeros((3, 1, 121, 3)), pred_world=np.zeros((3, 3, 1, 121, 3)))
    rmse, point, joint = evaluate_case(case, forecasts, calibrations, .2)
    assert all(r['prediction_RMSE_m'] is None and r['position_label_count'] == 0 for r in rmse)
    assert all(r['empirical_position_coverage'] is None for r in point)
    assert all(r['evaluation_scene_joint_covered'] is None for r in joint)
    agg_point, agg_joint = aggregate_coverage(point, joint)
    assert all(r['empirical_scene_joint_coverage'] is None for r in agg_joint)
    assert all(r['empirical_scene_horizon_coverage'] is None for r in agg_point)


def test_registry_requires_independent_nine_pools_and_correct_phase(tmp_path):
    path = tmp_path/'registry.json'
    path.write_text(json.dumps(dict(schema='stage5_artifact_registry_v1', phase='pilot', calibration_libraries=[])))
    with pytest.raises(ValueError, match='nine'):
        load_registry(path, 'pilot')
    with pytest.raises(ValueError, match='phase'):
        load_registry(path, 'formal')


def test_diagnostics_reject_calibration_as_unseen_and_preserve_existing_output(tmp_path):
    with pytest.raises(ValueError, match='never calibration'):
        run_diagnostics('pilot', 'calibration', 'missing.json', tmp_path/'out', plot=False)
    existing = tmp_path/'existing'
    existing.mkdir()
    with pytest.raises(FileExistsError):
        run_diagnostics('pilot', 'dev', 'missing.json', existing, plot=False)
