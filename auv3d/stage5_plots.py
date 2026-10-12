"""Saved six-DOF risk comparisons with visible seabed and separate forecasts."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .stage4_plots import (_read_case, _validate_pair, _points, _exact_record,
                           _new_output)

COLORS = dict(zip(('A0','A1','A2','A3','A4','A5'),
                 ('#2769ae','#9b59b6','#178249','#d88a19','#777777','#d42c31')))


def plot_comparison(case_dirs, out, anchor_s=40.):
    """Read recorded executions, preserving early failures and missing anchors."""
    if not case_dirs or set(case_dirs)-set(COLORS):
        raise ValueError('case_dirs requires available A0-A5 episode directories')
    if not np.isfinite(anchor_s) or anchor_s < 0:
        raise ValueError('anchor_s must be finite and nonnegative')
    cases = {method:_read_case(path) for method,path in case_dirs.items()}
    _validate_pair(cases)
    first = next(iter(cases.values()))
    scene_id = first['summary'].get('scene_metadata',{}).get('scene_id')
    if any(case['summary'].get('scene_metadata',{}).get('scene_id') != scene_id
           for case in cases.values()):
        raise ValueError('method comparison requires identical registered geometry')
    out = _new_output(out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .terrain import HeightField
    from .visualization import _draw_terrain, _draw_obstacles, _set_ned_axes
    saved = first['terrain']
    terrain = HeightField(saved['xs'], saved['ys'], saved['depths'], 0.)
    goal = np.asarray(first['summary']['goal_ned_m'])
    available, common_snapshot = {}, None
    for method,case in cases.items():
        snapshot = _exact_record(case['obstacles'],anchor_s)
        if snapshot is not None:
            if common_snapshot is not None and any(not np.array_equal(snapshot[key],common_snapshot[key])
                                                    for key in ('ids','positions','radii')):
                raise ValueError('paired saved obstacle truth differs at comparison clock')
            common_snapshot = snapshot
        available[method] = _exact_record(case['plans'],anchor_s) is not None
    for show_predictions,filename in ((False,'actual_trajectories_3d.png'),
                                      (True,'predicted_and_actual_3d.png')):
        fig = plt.figure(figsize=(13,9))
        ax = fig.add_subplot(projection='3d')
        fig.subplots_adjust(left=.02,right=.87,bottom=.06,top=.82)
        _,deepest = _draw_terrain(ax,terrain)
        if common_snapshot is not None:
            _draw_obstacles(ax,common_snapshot)
        for method,case in cases.items():
            points = _points(case['states'])
            ax.plot(*points.T,color=COLORS[method],lw=2,
                label=f"{method} actual REMUS | {case['summary']['status']}")
            ax.scatter(*points[-1],marker='x',color=COLORS[method],s=40)
            if show_predictions and available[method]:
                plan = _exact_record(case['plans'],anchor_s)
                rows = np.asarray(plan['states'])
                ax.plot(*rows[:,1:4].T,'--',color=COLORS[method],lw=1.1,
                        label=f'{method} forecast from {anchor_s:g}s')
            emergency = [event for event in case['events'] if event['kind']=='emergency']
            if emergency:
                # Display first event; all events remain in the raw episode.
                ax.scatter(*emergency[0]['position'],marker='X',s=55,color=COLORS[method])
        ax.scatter(*_points(first['states'])[0],color='green',s=65,label='Start')
        ax.scatter(*goal,color='black',marker='*',s=160,label='Goal (5 m arrival)')
        _set_ned_axes(ax,(saved['xs'][0],saved['xs'][-1]),
                      (saved['ys'][0],saved['ys'][-1]),deepest)
        ax.set_zlabel('Depth positive down (m)',labelpad=8)
        ax.view_init(elev=23,azim=-62)
        ax.set_title(f"{first['summary']['scenario']} | {scene_id or first['summary']['seed']}\n"
                     f"Original REMUS 6-DOF execution; seabed and obstacles at {anchor_s:g}s\n" +
                     ('Dashed curves are independent forecasts; X marks first emergency'
                      if show_predictions else 'Solid curves are actual execution; X marks first emergency'))
        ax.legend(fontsize=8,ncol=2,loc='upper left',bbox_to_anchor=(0.,1.02))
        fig.savefig(out/filename,dpi=160,bbox_inches='tight')
        plt.close(fig)
    fig,axes = plt.subplots(3,1,figsize=(11,10),sharex=True,constrained_layout=True)
    for method,case in cases.items():
        axes[0].plot(case['clearance']['time_s'],case['clearance']['actual_interval_clearance_m'],
                     color=COLORS[method],label=method)
        axes[1].plot(case['errors']['time_s'],case['errors']['position_error_m'],color=COLORS[method])
        axes[2].plot(case['controls']['time_s'],case['controls']['ref_rpm'],color=COLORS[method])
    axes[0].axhline(1.,color='black',ls='--',label='Actual 1 m criterion')
    axes[0].axhline(0.,color='red',ls=':',label='Actual collision')
    axes[0].set_ylabel('Actual physical net gap (m)')
    axes[0].legend(ncol=4)
    axes[1].set_ylabel('Selected REMUS prediction\nvs actual position error (m)')
    axes[2].set(ylabel='Command RPM',xlabel='Physical simulation time (s)')
    for ax in axes: ax.grid(alpha=.25)
    fig.suptitle(first['summary']['scenario']+' | all outcomes retained; no forecast-to-task probability claim')
    fig.savefig(out/'clearance_execution_error_control.png',dpi=160)
    plt.close(fig)
    record = dict(scene_id=scene_id,anchor_s=anchor_s,available_forecasts=available,
                  sources={method:case['source'] for method,case in cases.items()},
                  actual_physical_margin_m=1.,arrival_radius_m=5.,future_truth_used_online=False)
    (out/'FIGURE_SOURCES.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    return record
