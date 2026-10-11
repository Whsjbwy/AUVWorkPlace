#!/usr/bin/env python3
"""Stage-1 CLI. Existing run_remus.py and its environment remain untouched."""
from __future__ import annotations
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from auv3d.config import ExperimentConfig

ROOT=Path(__file__).resolve().parent

def output_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='task',required=True)
    a=sub.add_parser('audit');a.add_argument('--out',default='results_stage1/parameter_audit.json')
    a=sub.add_parser('regression');a.add_argument('--out',default='results_stage1/regression');a.add_argument('--duration',type=float,default=200.)
    a=sub.add_parser('remus-characterize');a.add_argument('--out',default='results_stage1/remus_characterize');a.add_argument('--duration',type=float,default=100.)
    a=sub.add_parser('remus-route');a.add_argument('--out',default='results_stage1/remus_route');a.add_argument('--max-time',type=float,default=500.)
    a=sub.add_parser('simulate');a.add_argument('--scenario',choices=('empty3d','static3d','crossing3d','paper_random3d'),default='crossing3d')
    a.add_argument('--config',default=None);a.add_argument('--seed',type=int,default=None);a.add_argument('--max-time',type=float,default=None)
    a.add_argument('--out',default='results_stage1/crossing3d');a.add_argument('--checkpoint');a.add_argument('--calibration')
    a.add_argument('--no-plot',action='store_true');a.add_argument('--overwrite',action='store_true')
    a=sub.add_parser('dataset');a.add_argument('--out',default='results_stage1/data/smoke.npz');a.add_argument('--trajectories',type=int,default=48)
    a.add_argument('--seconds',type=int,default=180);a.add_argument('--dt',type=float,default=3.);a.add_argument('--seed',type=int,default=123);a.add_argument('--overwrite',action='store_true')
    a=sub.add_parser('train');a.add_argument('--data',default='results_stage1/data/smoke.npz');a.add_argument('--out',default='results_stage1/pplnm_smoke')
    a.add_argument('--epochs',type=int,default=12);a.add_argument('--device',default='cpu');a.add_argument('--seed',type=int,default=0)
    a.add_argument('--resume',action='store_true');a.add_argument('--chunk-epochs',type=int);a.add_argument('--threads',type=int,default=2);a.add_argument('--batch-size',type=int,default=256);a.add_argument('--overwrite',action='store_true')
    a=sub.add_parser('batch');a.add_argument('--out',default='results_stage1/batch');a.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);a.add_argument('--max-time',type=float,default=160.)
    a.add_argument('--config',default=None);a.add_argument('--overwrite',action='store_true')
    args=p.parse_args(argv)
    if args.task=='audit':
        from auv3d.remus import parameter_audit
        result=parameter_audit();output_json(args.out,result)
    elif args.task=='regression':
        from auv3d.remus import regression
        result,data=regression(args.duration);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        output_json(out/'summary.json',result);np.savetxt(out/'states.csv',data,delimiter=',',header='time,n,e,d,roll,pitch,yaw,u,v,w,p,q,r,actual_rudder,actual_stern,actual_rpm',comments='')
    elif args.task=='remus-characterize':
        from auv3d.remus import characterize
        result,series=characterize(args.duration);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        output_json(out/'summary.json',result)
        for name,data in series.items():
            np.savetxt(out/f'{name}.csv',data,delimiter=',',header='time,n,e,d,roll,pitch,yaw,u,v,w,p,q,r,actual_rudder,actual_stern,actual_rpm',comments='')
    elif args.task=='remus-route':
        from auv3d.remus import route_demo
        result,data,refs=route_demo(args.max_time);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        output_json(out/'summary.json',result);np.savetxt(out/'states.csv',data,delimiter=',',header='time,n,e,d,roll,pitch,yaw,u,v,w,p,q,r,actual_rudder,actual_stern,actual_rpm',comments='')
        np.savetxt(out/'references.csv',refs,delimiter=',',header='time,waypoint,n,e,d',comments='')
        from auv3d.visualization import plt
        fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(111,projection='3d');ax.plot(data[:,1],data[:,2],data[:,3],label='6-DOF execution')
        wp=np.asarray(result['waypoints']);ax.scatter(wp[:,0],wp[:,1],wp[:,2],marker='*',s=100,label='Waypoints')
        ax.set(xlabel='North (m)',ylabel='East (m)',zlabel='Depth, down (m)',title=f"REMUS 100 3D tracking | {result['status']}\nNot an obstacle-avoidance benchmark")
        ax.invert_zaxis();ax.legend();fig.tight_layout();fig.savefig(out/'trajectory_3d.png',dpi=150);plt.close(fig)
    elif args.task=='simulate':
        from auv3d.simulation import run_episode,save_episode
        from auv3d.prediction import PPLNMPredictor,SphericalCalibration
        cfg=ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
        if args.seed is not None:cfg=replace(cfg,seed=args.seed)
        if args.max_time is not None:cfg=replace(cfg,planner=replace(cfg.planner,max_time_s=args.max_time))
        predictor=PPLNMPredictor(args.checkpoint) if args.checkpoint else None
        calibration=SphericalCalibration.load(args.calibration) if args.calibration else None
        if predictor and abs(predictor.dt-cfg.prediction_dt_s)>1e-9:raise ValueError('checkpoint dt must match prediction_dt_s')
        episode=run_episode(cfg,args.scenario,predictor,calibration);save_episode(episode,args.out,args.overwrite,not args.no_plot);result=episode['summary']
    elif args.task=='dataset':
        from auv3d.data import generate_dataset
        result=generate_dataset(args.out,args.trajectories,args.seconds,args.dt,args.seed,args.overwrite)
    elif args.task=='train':
        from auv3d.train import train_predictor
        result=train_predictor(args.data,args.out,args.epochs,args.seed,args.device,args.batch_size,args.threads,args.overwrite,args.chunk_epochs,args.resume)
    else:
        from auv3d.simulation import run_episode,save_episode,SCENARIOS
        cfg=ExperimentConfig.load(args.config) if args.config else ExperimentConfig();summaries=[]
        for seed in args.seeds:
            c=replace(cfg,seed=seed,planner=replace(cfg.planner,max_time_s=args.max_time))
            for name in SCENARIOS:
                ep=run_episode(c,name);save_episode(ep,Path(args.out)/f'{name}_seed{seed}',args.overwrite,plot=False)
                summaries.append(ep['summary']);print(name,seed,ep['summary']['status'],flush=True)
        result={'claim':'development integration tests, NOT independent success-rate evidence','episodes':summaries}
        output_json(Path(args.out)/'batch_summary.json',result)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError) as e:
        print(f'ERROR: {e}',file=sys.stderr);raise SystemExit(2)
