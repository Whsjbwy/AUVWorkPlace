"""Load saved REMUS diagnostics for offline rendering, never controller restart."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .scenarios import make_stage2_scenario
from .stage2_config import Stage2Config
from .terrain import HeightField


def _read_table(path, width):
    lines = path.read_text(encoding='utf-8').splitlines()
    if not lines:
        raise ValueError(f'{path.name} is missing its column header')
    columns = lines[0].split(',')
    if len(columns) != width:
        raise ValueError(f'{path.name} must have {width} named columns')
    body = [line for line in lines[1:] if line.strip()]
    if not body:
        return np.empty((0, width)), columns
    rows = np.asarray([[float(value) for value in line.split(',')] for line in body], dtype=float)
    if rows.ndim != 2 or rows.shape[1] != width or not np.isfinite(rows).all():
        raise ValueError(f'{path.name} must contain finite {width}-column rows')
    if np.any(np.diff(rows[:, 0]) < 0):
        raise ValueError(f'{path.name} times must be nondecreasing')
    return rows, columns


def _snapshot_arrays(row):
    stamp = float(row['time_s'])
    positions = np.asarray(row['positions'], dtype=float).reshape(-1, 3)
    radii = np.asarray(row['radii'], dtype=float).reshape(-1)
    ids = np.asarray(row['ids'])
    if not np.isfinite(stamp) or len(ids) != len(radii) or len(positions) != len(radii):
        raise ValueError('invalid saved obstacle snapshot dimensions or time')
    if not np.isfinite(positions).all() or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError('saved obstacle positions and positive radii must be finite')
    return stamp, ids, positions, radii


def _same_values(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-10, atol=1e-8))


def _append_verified_terminal_truth(result, cfg, scene):
    """Recreate only deterministic obstacle truth and verify recorded samples.

    Vehicle states are read verbatim from disk. Recreating an obstacle schedule
    for a missing display timestamp does not restore the vehicle controller,
    rerun planning, or provide new evidence of collision safety.
    """
    duration = float(result['states'][-1, 0])
    terrain = result['terrain']
    if not all(_same_values(getattr(terrain, key), getattr(scene.terrain, key))
               for key in ('xs', 'ys', 'depths')):
        raise ValueError('saved terrain differs from the configured shared scene; terminal truth cannot be reconstructed')
    targets = []
    for row in result['obstacles']:
        stamp, ids, positions, radii = _snapshot_arrays(row)
        tick = round(stamp/cfg.physics_dt)
        if abs(tick*cfg.physics_dt-stamp) > 1e-8:
            raise ValueError('saved obstacle snapshot does not align with the configured physics interval')
        targets.append((tick, stamp, ids, positions, radii))
    terminal_tick = round(duration/cfg.physics_dt)
    if abs(terminal_tick*cfg.physics_dt-duration) > 1e-8:
        raise ValueError('actual endpoint time does not align with the saved physics interval')
    if any(tick < 0 or tick > terminal_tick for tick, *_ in targets):
        raise ValueError('saved obstacle timestamp lies outside the actual execution')
    by_tick = {tick: (stamp, ids, positions, radii) for tick, stamp, ids, positions, radii in targets}
    if len(by_tick) != len(targets):
        raise ValueError('duplicate saved obstacle snapshot time')
    checked = 0
    maximum_error = 0.
    for tick in range(terminal_tick+1):
        current = scene.obstacles.snapshot()
        if tick in by_tick:
            stamp, ids, positions, radii = by_tick[tick]
            if not np.array_equal(current.ids, ids) or not _same_values(current.positions, positions) or not _same_values(current.radii, radii):
                raise ValueError(f'configured obstacle truth differs from saved snapshot at t={stamp:g}s; refusing terminal reconstruction')
            if positions.size:
                maximum_error = max(maximum_error, float(np.max(np.abs(current.positions-positions))))
            checked += 1
        if tick < terminal_tick:
            scene.obstacles.step(cfg.physics_dt)
    final = scene.obstacles.snapshot()
    result['obstacles'].append({'time_s':duration, 'ids':final.ids.tolist(),
                                'positions':final.positions.tolist(), 'radii':final.radii.tolist()})
    return {
        'terminal_snapshot_added':True,
        'terminal_truth_source':'shared deterministic obstacle generator, advanced at saved physics_dt; every original obstacle snapshot numerically matched first',
        'verified_original_snapshot_count':checked,
        'maximum_original_snapshot_position_difference_m':maximum_error,
        'comparison_absolute_tolerance_m':1e-8,
        'comparison_relative_tolerance':1e-10,
    }


def load_saved_remus_episode(source):
    """Return (renderable result, manifest) from the saved experiment files.

    This is an offline artifact loader. Integral, reference-filter and complete
    vehicle state are not stored in these files and cannot be restarted here.
    Missing terminal obstacle truth is reconstructed only after all stored
    samples and terrain agree numerically with the same configured scenario.
    """
    source = Path(source).absolute()
    summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if summary.get('backend') != 'remus6dof':
        raise ValueError('offline REMUS loader requires an explicitly remus6dof result')
    cfg = Stage2Config.load(source/'config.json')
    result = {'summary':summary, 'config':cfg.to_dict()}
    for name, width, columns in (('states',16,'state_columns'), ('controls',16,'control_columns'),
                                  ('planning',9,'planning_columns'), ('prediction_errors',7,'error_columns')):
        result[name], result[columns] = _read_table(source/f'{name}.csv', width)
    states = result['states']
    if not len(states):
        raise ValueError('saved REMUS execution has no actual states')
    duration = float(states[-1, 0])
    if not np.isclose(duration, summary['duration_s'], rtol=0., atol=1e-8):
        raise ValueError('actual state endpoint time differs from saved summary duration')
    with np.load(source/'terrain.npz', allow_pickle=False) as stored:
        xs, ys, depths = (np.asarray(stored[key], dtype=float) for key in ('xs','ys','depths'))
    if len(xs) < 2 or len(ys) < 2 or depths.shape != (len(ys),len(xs)):
        raise ValueError('saved terrain grid has incompatible dimensions')
    if not np.isfinite(np.r_[xs, ys, depths.ravel()]).all() or np.any(np.diff(xs) <= 0) or np.any(np.diff(ys) <= 0):
        raise ValueError('saved terrain grid must be finite and strictly increasing')
    gx = np.diff(depths, axis=1)/np.diff(xs)[None,:]
    gy = np.diff(depths, axis=0)/np.diff(ys)[:,None]
    result['terrain'] = HeightField(xs,ys,depths,float(np.hypot(np.abs(gx).max(),np.abs(gy).max())))
    result['selected_plans'] = json.loads((source/'selected_plans.json').read_text(encoding='utf-8'))
    result['events'] = json.loads((source/'events.json').read_text(encoding='utf-8'))
    result['obstacles'] = json.loads((source/'obstacles.json').read_text(encoding='utf-8'))
    times = [_snapshot_arrays(row)[0] for row in result['obstacles']]
    if any(a >= b for a,b in zip(times,times[1:])) or any(t < -1e-9 or t > duration+1e-9 for t in times):
        raise ValueError('saved obstacle times must be ordered within actual execution')
    scene = None
    if 'goal_ned_m' in summary:
        result['goal'] = np.asarray(summary['goal_ned_m'], dtype=float)
        goal_source = 'saved summary.goal_ned_m'
    else:
        scene = make_stage2_scenario(summary['scenario'],cfg)
        result['goal'] = np.asarray(scene.goal, dtype=float)
        goal_source = 'shared configured scene goal; legacy summary omitted goal_ned_m'
    if result['goal'].shape != (3,) or not np.isfinite(result['goal']).all():
        raise ValueError('saved/reconstructed goal must be a finite NED point')
    manifest = {
        'source_directory':str(source), 'purpose':'offline re-render of recorded REMUS results',
        'vehicle_controller_state_restored':False,
        'new_closed_loop_simulation_executed':False,
        'new_collision_safety_validation_performed':False,
        'actual_trajectory_source':'saved states.csv; vehicle trajectory is never re-simulated',
        'planned_trajectory_source':'previously generated REMUS predictions from selected_plans.json; sparse display rows only',
        'goal_source':goal_source, 'actual_endpoint_time_s':duration,
        'actual_endpoint_ned_m':states[-1,1:4].tolist(),
        'original_snapshot_count':len(times),
        'original_last_snapshot_time_s':times[-1] if times else None,
        'terminal_snapshot_added':False,
    }
    if not times or abs(times[-1]-duration) > 1e-8:
        scene = scene or make_stage2_scenario(summary['scenario'],cfg)
        manifest.update(_append_verified_terminal_truth(result,cfg,scene))
    manifest['terminal_snapshot_time_s'] = float(result['obstacles'][-1]['time_s'])
    manifest['terminal_snapshot_time_matches_actual_endpoint'] = bool(np.isclose(
        manifest['terminal_snapshot_time_s'], duration, rtol=0., atol=1e-8))
    return result, manifest


def render_saved_remus_episode(source, out, *, animate=True, max_frames=60, fps=8):
    """Export to a new directory without modifying any saved experiment file."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; choose a new offline rendering directory')
    result, manifest = load_saved_remus_episode(source)
    from .stage2_visualization import plot_stage2_episode, _animation_times
    from .stage2_replay import export_stage2_replay
    outputs = plot_stage2_episode(result,out,animate=animate,max_frames=max_frames,fps=fps)
    outputs['interactive_replay'] = export_stage2_replay(result,out/'replay_3d.html')
    if animate:
        from PIL import Image
        with Image.open(outputs['animation']) as gif:
            manifest['gif_frame_count'] = gif.n_frames
            gif.seek(gif.n_frames-1)
            last_frame = out/'trajectory_last_frame.png'
            gif.convert('RGBA').save(last_frame)
            outputs['decoded_gif_last_frame'] = str(last_frame)
        stamps = _animation_times(result['states'],result['obstacles'],max_frames)
        manifest['gif_last_frame_simulation_time_s'] = float(stamps[-1])
        manifest['gif_last_frame_matches_actual_endpoint_time'] = bool(np.isclose(
            stamps[-1], result['states'][-1,0], rtol=0., atol=1e-8))
    manifest['outputs'] = outputs
    (out/'render_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return manifest
