"""Check numeric/visual acceptance outputs using their actual values and dimensions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image


def inspect(out):
    report = json.loads((out / 'acceptance_report.json').read_text(encoding='utf-8'))
    if not report['all_passed']:
        raise ValueError('acceptance command checks did not all pass')
    original = np.loadtxt(out / 'original_remus/simulation.csv', delimiter=',', skiprows=1)
    regression = np.loadtxt(out / 'regression/states.csv', delimiter=',', skiprows=1)
    assert original.shape == (10001, 19) and regression.shape == (10001, 16)
    assert np.isfinite(original).all() and np.isfinite(regression).all()
    # Compare physical state values from two execution interfaces, never file identities.
    np.testing.assert_allclose(original[:, :13], regression[:, :13], rtol=0, atol=1e-10)
    np.testing.assert_allclose(original[:, 16:19], regression[:, 13:16], rtol=0, atol=1e-10)
    scenes = []
    for predictor in ('ca', 'pplnm'):
        for scenario in ('empty3d', 'static3d', 'crossing3d', 'paper_random3d'):
            directory = out / f'{predictor}_{scenario}'
            summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
            cfg = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
            states = np.loadtxt(directory / 'trajectory.csv', delimiter=',', skiprows=1)
            plans = json.loads((directory / 'selected_plans.json').read_text(encoding='utf-8'))
            assert states.ndim == 2 and states.shape[1] == 10 and np.isfinite(states).all()
            assert np.all(np.diff(states[:, 0]) > 0)
            assert summary['status'] == 'success' and summary['failure_reason'] is None
            assert summary['goal_distance_m'] <= cfg['planner']['goal_tolerance_m']
            assert summary['minimum_model_clearance_m'] > 0
            with np.load(directory / 'terrain.npz', allow_pickle=False) as terrain:
                depths = terrain['depths']
                assert np.isfinite(depths).all() and np.min(depths) > 0
                depth_range = [float(np.min(depths)), float(np.max(depths))]
            for plan in plans:
                prediction = np.asarray(plan['states'])
                assert prediction.shape == (46, 6) and np.isfinite(prediction).all()
                index = np.argmin(abs(states[:, 0] - plan['time_s']))
                assert abs(states[index, 0] - plan['time_s']) < 1e-9
                np.testing.assert_allclose(prediction[0, :3], states[index, 1:4], rtol=0, atol=1e-9)
            for name in ('terrain_3d.png', 'trajectory_3d.png', 'trajectory_local_3d.png', 'depth.png', 'speed.png'):
                with Image.open(directory / name) as picture:
                    picture.load()
                    assert min(picture.size) > 100
            scenes.append({'name': directory.name, 'status': summary['status'],
                           'samples': len(states), 'selected_plans': len(plans),
                           'terrain_depth_range_m': depth_range, 'images_readable': 5})
    with Image.open(out / 'original_remus/trajectory.gif') as animation:
        frames = animation.n_frames
        assert frames > 1
        for index in range(frames):
            animation.seek(index)
            animation.load()
    cases = list(ET.parse(out / 'pytest.xml').iter('testcase'))
    original_tests = sum(c.attrib.get('classname', '').startswith('tests.') for c in cases)
    return {'all_passed': True, 'original_regression_max_state_difference': float(np.max(abs(original[:, :13]-regression[:, :13]))),
            'remus_animation_frames': frames, 'original_tests': original_tests,
            'stage1_tests': len(cases)-original_tests, 'scenes': scenes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_directory', type=Path)
    args = parser.parse_args()
    result = inspect(args.result_directory)
    text = json.dumps(result, indent=2)
    (args.result_directory / 'numeric_visual_checks.json').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
