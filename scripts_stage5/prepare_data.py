"""Register new scenes and acquire independent causal calibration libraries."""
import argparse
import json
from pathlib import Path
import time

from auv3d.stage5_protocol import manifest, make_scene, serialize
from auv3d.stage5_calibration import build_libraries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('pilot', 'formal'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--obstacle-only', action='store_true',
        help='Use all four obstacle families for risk calibration; all terrain cases remain registered.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    serialize(args.out/'PREREGISTRATION.json', args.phase)
    rows = manifest(args.phase)
    preflight = []
    # One noise replica suffices for geometry; noise never changes the map.
    for row in manifest(args.phase, noise_std=.02):
        try:
            scene = make_scene(row)
            preflight.append(dict(case_id=row['case_id'], valid=True,
                obstacle_count=len(scene.obstacles.snapshot().ids),
                scene_metadata=scene.metadata))
        except Exception as error:
            preflight.append(dict(case_id=row['case_id'], split=row['split'],
                                  family=row['family'], valid=False,
                                  error=f'{type(error).__name__}: {error}'))
    (args.out/'PREFLIGHT.json').write_text(json.dumps(preflight, indent=2,
        ensure_ascii=False, allow_nan=False), encoding='utf-8')
    blocking = [row for row in preflight if not row['valid'] and not (
        args.obstacle_only and row['split'] == 'calibration' and
        row['family'] in ('terrain3d', 'perlin_cruise_constrained3d'))]
    if blocking:
        raise ValueError('registered scene factory failure; preserve IDs and do not retry seeds')
    print(f"{args.phase}: {sum(row['valid'] for row in preflight)}/{len(preflight)} geometries valid; all original IDs retained", flush=True)
    if args.obstacle_only:
        # Empty terrain scenes cannot supply obstacle error labels. Exclude
        # both terrain families as a whole, rather than retrying a failed seed.
        rows = [row for row in rows if row['family'] not in
                ('terrain3d', 'perlin_cruise_constrained3d')]
    started = time.perf_counter()
    libraries = build_libraries(rows, args.out/'libraries')
    inventory = []
    for row in libraries:
        meta = row['metadata']
        inventory.append({key: value for key, value in row.items() if key != 'metadata'} | {
            key: meta[key] for key in ('phase', 'active_scene_count', 'empty_scene_count',
                                       'observed_speed_bin_scene_counts')})
        print(inventory[-1], flush=True)
    (args.out/'CALIBRATION_INVENTORY.json').write_text(json.dumps(dict(
        elapsed_s=time.perf_counter()-started, libraries=inventory), indent=2,
        ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (args.out/'ARTIFACT_REGISTRY.json').write_text(json.dumps(dict(
        schema='stage5_artifact_registry_v1', phase=args.phase,
        calibration_libraries=[dict(predictor=row['predictor'], noise_std_m=row['noise_std_m'],
            path=str(Path(row['path']).absolute())) for row in inventory]),
        indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
