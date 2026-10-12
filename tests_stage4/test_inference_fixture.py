"""The distributed small library loads without local paths or test labels."""
from pathlib import Path

import numpy as np
import pytest

from auv3d.stage4_protocol import manifest
from auv3d.stage4_risk import RiskPolicy


@pytest.mark.parametrize('method', ['B1', 'B2'])
@pytest.mark.parametrize('alpha', [.1, .2, .3])
def test_distributed_library_matches_registered_calibration_contract(method, alpha):
    path = Path(__file__).resolve().parents[1]/'artifacts_stage4'/'ca_scene_calibration_v1.npz'
    policy = RiskPolicy(method, path, alpha)
    metadata = policy.metadata
    expected = {row['case_id'] for row in manifest() if row['split'] == 'calibration'}
    assert set(metadata['calibration_case_ids']) == expected
    assert metadata['active_calibration_scene_count'] == 16
    assert metadata['empty_calibration_scene_count'] == 8
    assert metadata['calibration_horizon_s'] == 24.
    assert np.isfinite(metadata['ball_quantile_m'])
    for scene in metadata['library_parameters']['scenes']:
        assert not Path(scene['source_directory']).is_absolute()
        assert scene['case_id'] in expected
    sampled_path = np.column_stack([policy.times, np.zeros((len(policy.times), 2))])
    from auv3d.stage4_risk import ecp_scene_scores, upper_quantile
    q = upper_quantile(ecp_scene_scores(sampled_path, policy.pred, policy.labels,
                                      policy.radii, policy.mask, .8056208785775106), alpha)
    assert 0 <= q <= policy.ball_q + 1e-9
