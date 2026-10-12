"""Render recorded physical data; never run a new mission or replace raw files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from auv3d.stage4_plots import _read_case, _new_output
from auv3d.stage2_simulation import STATE_COLUMNS, CONTROL_COLUMNS, ERROR_COLUMNS
from auv3d.terrain import HeightField


def load_saved_result(directory):
    saved = _read_case(directory)
    terrain = saved['terrain']
    # Actual injected predictor, rather than the legacy config default, labels
    # this display. The archived config is never written back or overwritten.
    display_config = dict(saved['config'], obstacle_predictor=saved['summary']['obstacle_predictor'])
    return dict(summary=saved['summary'], config=display_config,
        terrain=HeightField(terrain['xs'], terrain['ys'], terrain['depths'], 0.),
        goal=np.asarray(saved['summary']['goal_ned_m']),
        states=np.column_stack([saved['states'][name] for name in STATE_COLUMNS]),
        controls=np.column_stack([saved['controls'][name] for name in CONTROL_COLUMNS]),
        control_columns=CONTROL_COLUMNS,
        prediction_errors=np.column_stack([saved['errors'][name] for name in ERROR_COLUMNS]),
        error_columns=ERROR_COLUMNS, selected_plans=saved['plans'],
        obstacles=saved['obstacles'], events=saved['events'])


def render_saved(directory, out, *, animate=True, frames=40):
    from auv3d.stage2_visualization import plot_stage2_episode
    from auv3d.stage2_replay import export_stage2_replay
    result = load_saved_result(directory)
    out = _new_output(out)
    paths = plot_stage2_episode(result, out, animate=animate, max_frames=frames)
    paths['replay'] = export_stage2_replay(result, out/'replay_3d.html')
    record = dict(source=str(Path(directory).resolve()),
        source_method=result['summary'].get('risk_method'),
        actual_predictor=result['summary']['obstacle_predictor'],
        states_original_count=len(result['states']),
        display_only=True, new_simulation_executed=False, outputs=paths)
    (out/'FIGURE_SOURCES.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--frames', type=int, default=40)
    parser.add_argument('--no-animation', action='store_true')
    args = parser.parse_args()
    print(json.dumps(render_saved(args.case,args.out,animate=not args.no_animation,frames=args.frames),indent=2,ensure_ascii=False))
