"""Re-render saved REMUS results offline; this does not resume a controller."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).absolute().parents[1]))
from auv3d.stage2_outputs import render_saved_remus_episode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,help='Saved REMUS result directory')
    parser.add_argument('--out',required=True,help='New or empty offline rendering directory')
    parser.add_argument('--max-frames',type=int,default=60)
    parser.add_argument('--fps',type=float,default=8.)
    parser.add_argument('--no-animation',action='store_true')
    args = parser.parse_args(argv)
    manifest = render_saved_remus_episode(args.source,args.out,animate=not args.no_animation,
                                           max_frames=args.max_frames,fps=args.fps)
    print(json.dumps(manifest,indent=2,ensure_ascii=False))
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,RuntimeError,FileExistsError,FileNotFoundError) as error:
        print(f'ERROR: {error}')
        raise SystemExit(2)
