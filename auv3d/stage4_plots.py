"""Offline paired research figures from saved REMUS runs, never online truth."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


COLORS = {'B0': '#2769ae', 'B1': '#e38a20', 'B2': '#178249'}
TERRAIN_FAMILIES = {'terrain3d', 'perlin_cruise_constrained3d'}


def _new_output(path):
    out = Path(path)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'preserve existing research figures: {out}')
    out.mkdir(parents=True, exist_ok=True)
    return out


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _table(path, required):
    values = np.genfromtxt(path, names=True, delimiter=',', dtype=float, encoding='utf-8')
    if values.dtype.names is None or any(name not in values.dtype.names for name in required):
        raise ValueError(f'{path} is missing required named columns')
    table = {name: np.atleast_1d(values[name]) for name in values.dtype.names}
    if any(not np.isfinite(column).all() for column in table.values()):
        raise ValueError(f'{path} contains nonfinite numeric values')
    if 'time_s' in table and np.any(np.diff(table['time_s']) <= 0):
        raise ValueError(f'{path} time_s must be strictly increasing')
    return table


def _read_case(directory):
    directory = Path(directory)
    summary, cfg = _json(directory / 'summary.json'), _json(directory / 'config.json')
    if summary.get('backend') != 'remus6dof':
        raise ValueError('paired risk figures require actual REMUS six-DOF results')
    states = _table(directory / 'states.csv', ('time_s', 'north_m', 'east_m', 'depth_m'))
    if not len(states['time_s']):
        raise ValueError('actual trajectory must contain at least one state')
    controls = _table(directory / 'controls.csv', ('time_s', 'ref_rpm', 'actual_rpm'))
    clearance = _table(directory / 'clearance.csv', ('time_s', 'actual_interval_clearance_m'))
    errors = _table(directory / 'prediction_errors.csv', ('time_s', 'position_error_m'))
    with np.load(directory / 'terrain.npz', allow_pickle=False) as data:
        terrain = {name: np.asarray(data[name], dtype=float) for name in ('xs', 'ys', 'depths')}
    if (terrain['xs'].ndim != 1 or terrain['ys'].ndim != 1
            or terrain['depths'].shape != (len(terrain['ys']), len(terrain['xs']))
            or any(not np.isfinite(array).all() for array in terrain.values())
            or np.any(np.diff(terrain['xs']) <= 0) or np.any(np.diff(terrain['ys']) <= 0)):
        raise ValueError('saved terrain requires finite ordered axes and matching depths')
    return {'source': str(directory.resolve()), 'summary': summary, 'config': cfg,
            'states': states, 'controls': controls, 'clearance': clearance, 'errors': errors,
            'terrain': terrain, 'plans': _json(directory / 'selected_plans.json'),
            'diagnostics': _json(directory / 'planning_diagnostics.json'),
            'obstacles': _json(directory / 'obstacles.json'), 'events': _json(directory / 'events.json')}


def _validate_pair(cases):
    first = next(iter(cases.values()))
    ignored = {'risk_method', 'risk_alpha', 'risk_library', 'label', '_provenance'}
    reference = {key: value for key, value in first['config'].items() if key not in ignored}
    for case in cases.values():
        if any(case['summary'].get(key) != first['summary'].get(key) for key in ('scenario', 'seed')):
            raise ValueError('paired plots must use the same scenario and random seed')
        cfg = {key: value for key, value in case['config'].items() if key not in ignored}
        if cfg != reference:
            raise ValueError('paired plots require identical actual common configuration fields')
        if any(not np.array_equal(case['terrain'][key], first['terrain'][key])
               for key in ('xs', 'ys', 'depths')):
            raise ValueError('paired terrain arrays differ')
        if not np.array_equal(case['summary']['goal_ned_m'], first['summary']['goal_ned_m']):
            raise ValueError('paired final goals differ')


def _points(table):
    return np.column_stack([table[name] for name in ('north_m', 'east_m', 'depth_m')])


def _at_time(table, time):
    if not table['time_s'][0] <= time <= table['time_s'][-1]:
        return None
    return np.array([np.interp(time, table['time_s'], table[name])
                     for name in ('north_m', 'east_m', 'depth_m')])


def _exact_record(records, time):
    selected = [row for row in records if abs(float(row['time_s']) - time) < 1e-7]
    if len(selected) > 1:
        raise ValueError('duplicate saved record at the requested comparison timestamp')
    return selected[0] if selected else None


def _risk_trace(diagnostics, method):
    times, lower, upper = [], [], []
    for plan in diagnostics:
        values = [float(row['quantile_m']) for row in plan.get('risk_records', [])
                  if row.get('action_kind') == 'candidate']
        if not values and method == 'B0':
            values = [0.]
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError('saved risk quantiles must be finite and nonnegative')
        times.append(float(plan['time_s']))
        lower.append(min(values) if values else np.nan)
        upper.append(max(values) if values else np.nan)
    return np.asarray(times), np.asarray(lower), np.asarray(upper)


def plot_paired_case(case_dirs, out, anchor_s=40.):
    """Plot available B0/B1/B2 actual traces and separate same-time forecasts.

    A missing anchor after an early failure is reported; no nearby timestamp or
    continued ideal trajectory is substituted. All scene/config/terrain fields
    are compared directly. This diagnostic reads only already saved files.
    """
    if not case_dirs or any(method not in COLORS for method in case_dirs):
        raise ValueError('case_dirs must map available B0/B1/B2 methods to episode directories')
    if not np.isfinite(anchor_s) or anchor_s < 0:
        raise ValueError('comparison anchor must be finite and nonnegative')
    cases = {method: _read_case(directory) for method, directory in case_dirs.items()}
    _validate_pair(cases)
    out = _new_output(out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .terrain import HeightField
    from .visualization import _draw_terrain, _draw_obstacles, _set_ned_axes, _finish_3d

    first = next(iter(cases.values()))
    saved = first['terrain']
    terrain = HeightField(saved['xs'], saved['ys'], saved['depths'], 0.)
    goal = np.asarray(first['summary']['goal_ned_m'], dtype=float)
    trajectories = np.concatenate([_points(case['states']) for case in cases.values()])
    low = np.minimum(trajectories.min(axis=0), goal) - 18.
    high = np.maximum(trajectories.max(axis=0), goal) + 18.
    nlim = (max(float(saved['xs'][0]), low[0]), min(float(saved['xs'][-1]), high[0]))
    elim = (max(float(saved['ys'][0]), low[1]), min(float(saved['ys'][-1]), high[1]))
    if nlim[0] >= nlim[1] or elim[0] >= elim[1]:
        raise ValueError('saved actual trajectories fall outside their common terrain domain')
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    surface, deepest = _draw_terrain(ax, terrain, nlim, elim)
    snapshot = None
    availability = {}
    for method, case in cases.items():
        points, table = _points(case['states']), case['states']
        status = case['summary']['status']
        ax.plot(*points.T, color=COLORS[method], linewidth=2.1,
                label=f'{method} REMUS actual | {status}')
        ax.scatter(*points[-1], marker='x', color=COLORS[method], s=35, depthshade=False)
        position = _at_time(table, anchor_s)
        if position is not None:
            ax.scatter(*position, marker='o', color=COLORS[method], edgecolors='black', s=45,
                       depthshade=False, label=f'{method} actual at {anchor_s:g}s')
        plan = _exact_record(case['plans'], anchor_s)
        if plan is not None:
            rows = np.asarray(plan['states'], dtype=float)
            if rows.ndim != 2 or rows.shape[1] != 16 or not np.isfinite(rows).all():
                raise ValueError('saved predictions require finite [offset,eta6,nu6,actuator3] states')
            ax.plot(*rows[:, 1:4].T, '--', color=COLORS[method], linewidth=1.5,
                    label=f'{method} predicted from {anchor_s:g}s (not execution)')
        obstacle = _exact_record(case['obstacles'], anchor_s)
        if obstacle is not None:
            if snapshot is not None and any(not np.array_equal(obstacle[key], snapshot[key])
                                            for key in ('ids', 'positions', 'radii')):
                raise ValueError('same-time obstacle truth differs across paired methods')
            snapshot = obstacle
        marked = False
        for event in case['events']:
            if 'emergency' not in str(event.get('kind', '')) and 'collision' not in str(event.get('kind', '')):
                continue
            event_position = np.asarray(event.get('position', _at_time(table, float(event['time_s']))), float)
            if event_position.shape != (3,) or not np.isfinite(event_position).all():
                raise ValueError('saved failure/emergency event needs a finite actual position')
            ax.scatter(*event_position, marker='X', color=COLORS[method], s=35, alpha=.65,
                       label=f'{method} emergency / failure' if not marked else None, depthshade=False)
            marked = True
        availability[method] = {'actual_at_anchor': position is not None,
                                'prediction_at_anchor': plan is not None,
                                'obstacle_snapshot_at_anchor': obstacle is not None,
                                'status': status, 'duration_s': float(table['time_s'][-1])}
    if snapshot is not None:
        _draw_obstacles(ax, snapshot)
    start = _points(first['states'])[0]
    ax.scatter(*start, marker='o', color='#54ba69', edgecolors='black', s=65, label='Common start', depthshade=False)
    ax.scatter(*goal, marker='*', color='#ce2737', edgecolors='black', s=140, label='Common goal', depthshade=False)
    _set_ned_axes(ax, nlim, elim, max(deepest, float(trajectories[:, 2].max()), float(goal[2])))
    missing = ', '.join(method for method, info in availability.items() if not info['prediction_at_anchor'])
    ax.set_title(f"{first['summary']['scenario']} | seed {first['summary']['seed']} | paired risk pilot\n"
                 f'REMUS actual execution (solid) and forecasts from t={anchor_s:g}s (dashed)\n'
                 + (f'No saved prediction at anchor: {missing}' if missing else 'Same timestamp for vehicle markers and obstacle snapshot'))
    _finish_3d(fig, ax, surface, out / 'paired_actual_3d.png')

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for method, case in cases.items():
        color = COLORS[method]
        axes[0].plot(case['clearance']['time_s'], case['clearance']['actual_interval_clearance_m'],
                     color=color, label=method)
        times, lower, upper = _risk_trace(case['diagnostics'], method)
        axes[1].plot(times, upper, color=color, label=f'{method} candidate maximum q')
        axes[1].fill_between(times, lower, upper, color=color, alpha=.15)
        axes[2].plot(case['controls']['time_s'], case['controls']['actual_rpm'], color=color,
                     label=f'{method} actual RPM')
        axes[2].plot(case['controls']['time_s'], case['controls']['ref_rpm'], '--', color=color,
                     alpha=.7, label=f'{method} RPM reference')
        axes[3].plot(case['errors']['time_s'], case['errors']['position_error_m'], color=color, label=method)
    axes[0].axhline(1., color='black', linestyle='--', label='Unchanged 1 m safety threshold')
    axes[0].axhline(0., color='#ba2538', linestyle=':', label='Collision boundary')
    titles = ('Actual synchronous swept clearance', 'Candidate q range per plan; not selected-action q',
              'Reference and actual propeller response', 'Selected-plan vs actual position error')
    for axis, title, unit in zip(axes, titles, ('Clearance (m)', 'Added risk q (m)', 'RPM', 'Position error (m)')):
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel('Simulation time (s)')
    statuses = ' | '.join(f"{method}: {case['summary']['status']}, min={case['summary']['minimum_model_clearance_m']:.3f}m"
                          for method, case in cases.items())
    fig.suptitle(f"{first['summary']['scenario']} | seed {first['summary']['seed']}\n{statuses}", fontsize=11)
    fig.tight_layout(rect=(0., 0., 1., .95))
    fig.savefig(out / 'paired_risk_controls_error.png', dpi=150)
    plt.close(fig)
    metadata = {'scenario': first['summary']['scenario'], 'seed': first['summary']['seed'],
                'anchor_s': float(anchor_s), 'methods': list(cases), 'anchor_availability': availability,
                'sources': {method: case['source'] for method, case in cases.items()},
                'actual_parameters_compared_directly': True, 'terrain_arrays_compared_directly': True,
                'online_truth_used': False, 'quantile_plot_scope': 'range over verified candidates, not selected q'}
    (out / 'FIGURE_METADATA.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def _number(value):
    if value in (None, '', 'None', 'null'):
        return None
    if str(value).lower() in ('true', 'false'):
        return float(str(value).lower() == 'true')
    result = float(value)
    return result if np.isfinite(result) else None


def _metric_group(metric, rows):
    values = [_number(row[metric]) for row in rows]
    known = [value for value in values if value is not None]
    if metric == 'navigation_success':
        # A program error or absent outcome is not evidence of reaching the
        # goal. Keep every registered result row in the arrival denominator.
        value = sum(value == 1. for value in values) / len(rows) if rows else None
        aggregation = 'confirmed_arrivals_over_all_registered_results'
    else:
        value = float(np.median(known) if metric == 'planning_P95_ms' else np.mean(known)) if known else None
        aggregation = 'median_episode_P95' if metric == 'planning_P95_ms' else 'mean_known_episode_indicator'
    return value, len(known), aggregation


def plot_risk_curves(csv_path, out):
    """Separate every split and terrain negative controls; never pool groups."""
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'split', 'family', 'risk_method', 'risk_alpha', 'navigation_success', 'collision',
                'actual_clearance_margin_violation', 'planning_P95_ms', 'case_id'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('risk curves require the named unified experiment metrics')
    if any(row['split'] not in ('train', 'tune', 'calibration', 'test')
           or row['risk_method'] not in COLORS for row in rows):
        raise ValueError('unknown registered split or risk method')
    out = _new_output(out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = [('navigation_success', 'Time-limited arrival fraction'),
               ('collision', 'Actual collision fraction'),
               ('actual_clearance_margin_violation', 'Actual 1 m violation fraction'),
               ('planning_P95_ms', 'Median of episode planning P95 (ms)')]
    summaries, figures = [], []
    for split in ('train', 'tune', 'calibration', 'test'):
        selected = [row for row in rows if row['split'] == split]
        if not selected:
            continue
        fig, axes = plt.subplots(4, 2, figsize=(13, 13), squeeze=False)
        for column, (scope, negative) in enumerate((('obstacle_scenes', False), ('terrain_negative_control', True))):
            scoped = [row for row in selected if (row['family'] in TERRAIN_FAMILIES) == negative]
            for index, (metric, title) in enumerate(metrics):
                ax = axes[index, column]
                for method in COLORS:
                    method_rows = [row for row in scoped if row['risk_method'] == method]
                    levels = sorted({_number(row['risk_alpha']) for row in method_rows
                                     if _number(row['risk_alpha']) is not None}) if method != 'B0' else [None]
                    xs, ys = [], []
                    for level in levels:
                        group = [row for row in method_rows if method == 'B0' or _number(row['risk_alpha']) == level]
                        if len({row['case_id'] for row in group}) != len(group):
                            raise ValueError('duplicate scenario rows in one split/method/alpha group')
                        value, measured, aggregation = _metric_group(metric, group)
                        summaries.append({'split': split, 'scope': scope, 'method': method,
                                          'alpha': level, 'metric': metric, 'registered_count': len(group),
                                          'measured_count': measured, 'value': value,
                                          'aggregation': aggregation})
                        if value is None:
                            continue
                        if method == 'B0':
                            ax.axhline(value, color=COLORS[method], linestyle='--',
                                       label=f'B0 fixed margin (known={measured}/{len(group)})')
                        else:
                            xs.append(level)
                            ys.append(value)
                    if xs:
                        ax.plot(xs, ys, 'o-', color=COLORS[method], label=method)
                ax.set_title(f'{scope} | {title}')
                ax.set_xlabel('Predeclared risk alpha (B0 has no alpha)')
                ax.grid(alpha=.25)
                handles, _ = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=8)
                if metric != 'planning_P95_ms':
                    ax.set_ylim(-.03, 1.03)
                if not scoped:
                    ax.text(.5, .5, 'No registered results in this group', transform=ax.transAxes, ha='center')
        fig.suptitle(f'{split} split only | paired risk pilot\n'
                     'Arrival denominator includes all result rows; other unknown metrics retain known/registered counts', fontsize=11)
        fig.tight_layout(rect=(0., 0., 1., .95))
        path = out / f'risk_curves_{split}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(str(path))
    with (out / 'CURVE_SUMMARY.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('split', 'scope', 'method', 'alpha', 'metric',
                                                   'registered_count', 'measured_count', 'value', 'aggregation'))
        writer.writeheader()
        writer.writerows(summaries)
    return {'figures': figures, 'groups': summaries,
            'scope': 'each split and obstacle/terrain category kept separate; episode planning P95 medians are not pooled P95'}
