"""Fixed first-failure, same-state offline candidate diagnostics.

Only the first planning call with no feasible shortlisted reference is captured.
The rule is fixed before execution and does not select a favorable outcome.
The observer has no command authority. Its offline full-grid work starts after
the episode, is timed separately, and cannot affect measured planning latency.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from auv3d.adapters.remus_guidance import clone_plant
from auv3d.remus_planner import evaluate_reference_grid
from auv3d.stage2_failure_analysis import captured_plant_record


class FirstFailureCapture:
    def __init__(self, cfg):
        self.cfg = cfg
        self.previous_action = None
        self.frame = None

    def __call__(self, plant, tracks, now, plan, scene):
        if self.frame is None and not plan.feasible:
            uncertainty = {}
            for record in plan.forecast_records:
                object_id = int(record['object_id'])
                value = float(record['radius_components']['memory_extra_uncertainty_m'])
                if object_id in uncertainty and uncertainty[object_id] != value:
                    raise ValueError('one plan must use consistent memory uncertainty per object')
                uncertainty[object_id] = value
            if any(int(track.object_id) not in uncertainty for track in tracks):
                raise ValueError('captured tracks must have the applied uncertainty metadata')
            self.frame = {'plant': clone_plant(plant), 'tracks': deepcopy(list(tracks)),
                          'now': float(now), 'previous_action': deepcopy(self.previous_action),
                          'selected_action': deepcopy(plan.action),
                          'planning_goal': np.asarray(plan.planning_goal, dtype=float).copy(),
                          'final_goal': np.asarray(scene.goal, dtype=float).copy(),
                          'terrain': deepcopy(scene.terrain), 'uncertainty_by_id': uncertainty,
                          'selected_reason': str(plan.reason),
                          'selected_emergency_safe': bool(plan.emergency_verified_safe),
                          'planning_ms': float(plan.planning_ms)}
        self.previous_action = deepcopy(plan.action)

    def evaluate(self, output):
        out = Path(output)
        rule = 'first planning call with feasible=False; at most one frame per case'
        if self.frame is None:
            result = {'capture_rule': rule, 'captured': False, 'checked_frame_count': 0,
                      'offline_wall_clock_s': 0.,
                      'scope': 'no failed planning call captured; this is not an exhaustive omission audit'}
        else:
            frame = self.frame
            tic = time.perf_counter()
            grid = evaluate_reference_grid(frame['plant'], frame['planning_goal'], frame['terrain'],
                frame['tracks'], frame['now'], self.cfg, previous_action=frame['previous_action'],
                uncertainty_by_id=frame['uncertainty_by_id'])
            selected_count = grid['shortlist_feasible_counts'][str(self.cfg.candidate_budget)]
            result = {'capture_rule': rule, 'captured': True, 'checked_frame_count': 1,
                      'time_s': frame['now'],
                      'scope': 'same original REMUS/controller/reference/actuator state and causal forecast; first-failure only, not all failed calls',
                      'offline_wall_clock_s': time.perf_counter()-tic,
                      'timing_scope': 'after episode; excluded from online planning_ms and deadline counts',
                      'captured_plant': captured_plant_record(frame['plant']),
                      'previous_action': (frame['previous_action'].as_array().tolist()
                                          if frame['previous_action'] is not None else None),
                      'selected_action': frame['selected_action'].as_array().tolist(),
                      'planning_goal_ned_m': frame['planning_goal'].tolist(),
                      'final_task_goal_ned_m': frame['final_goal'].tolist(),
                      'uncertainty_by_id': frame['uncertainty_by_id'],
                      'selected_reason': frame['selected_reason'],
                      'selected_emergency_safe': frame['selected_emergency_safe'],
                      'selected_planning_ms': frame['planning_ms'],
                      'tracks': [{'object_id': int(track.object_id), 'radius_m': float(track.radius_m),
                                  'times_s': np.asarray(track.times).tolist(),
                                  'positions': np.asarray(track.positions).tolist()}
                                 for track in frame['tracks']],
                      'full_grid_safe_count': int(grid['feasible_count']),
                      'configured_shortlist_safe_count': int(selected_count),
                      'configured_shortlist_omitted_safe_reference': bool(grid['feasible_count'] > 0 and selected_count == 0),
                      'evaluation_source': 'independent full production grid rollout after execution',
                      'grid': grid}
        (out/'candidate_first_failure_diagnostic.json').write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        return {key: result.get(key) for key in (
            'capture_rule', 'captured', 'checked_frame_count', 'time_s', 'scope',
            'offline_wall_clock_s', 'full_grid_safe_count', 'configured_shortlist_safe_count',
            'configured_shortlist_omitted_safe_reference')}
