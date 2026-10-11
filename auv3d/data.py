from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json, time
import numpy as np
from .config import ObstacleConfig
from .terrain import HeightField
from .obstacles import PIObstacles
from .prediction import make_history,constant_acceleration

def generate_dataset(out,n_trajectories=48,seconds=80,dt=1.,seed=123,overwrite=False):
    out=Path(out)
    if out.exists() and not overwrite:raise FileExistsError(out)
    if n_trajectories<20 or seconds<12 or dt<=0:raise ValueError('dataset too small or invalid dt')
    frames=int(seconds/dt)+1;substeps=max(1,int(np.ceil(dt/.2)));subdt=dt/substeps
    X=[];Y=[];P=[];ids=[];speeds=[];boundary_rejected=0
    tic=time.perf_counter()
    for traj in range(n_trajectories):
        field=PIObstacles(replace(ObstacleConfig(),count=1),HeightField.flat(),seed+traj)
        pos=[];flags=[];previous=field.boundary_events
        for f in range(frames):
            pos.append(field.states[0,:3].copy());flags.append(field.boundary_events>previous);previous=field.boundary_events
            for _ in range(substeps):field.step(subdt)
        pos=np.asarray(pos);times=np.arange(frames)*dt
        for t in range(5,frames-3):
            if any(flags[t-5:t+4]):boundary_rejected+=1;continue
            hist=make_history(times[t-5:t+1],pos[t-5:t+1],dt)
            X.append(hist.features);Y.append(pos[t+1:t+4]);P.append(constant_acceleration(hist.features,np.arange(1,4)*dt))
            ids.append(traj);speeds.append(np.linalg.norm(hist.features[-1,3:6]))
    # Split complete independent trajectories, never overlapping windows.
    rng=np.random.default_rng(seed+90000);order=rng.permutation(n_trajectories)
    ntrain=int(.8*n_trajectories);remaining=n_trajectories-ntrain
    nval=max(1,remaining//3);ncal=max(1,(remaining-nval)//2)
    group=np.empty(n_trajectories,dtype=np.int64)
    group[order[:ntrain]]=0;group[order[ntrain:ntrain+nval]]=1
    group[order[ntrain+nval:ntrain+nval+ncal]]=2;group[order[ntrain+nval+ncal:]]=3
    ids=np.asarray(ids);split=group[ids]
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,X=np.asarray(X,dtype=np.float32),Y=np.asarray(Y,dtype=np.float32),
                        physical=np.asarray(P,dtype=np.float32),trajectory_id=ids,split=split,
                        speed=np.asarray(speeds,dtype=np.float32),dt=np.array(dt))
    meta={'n_trajectories':n_trajectories,'seconds':seconds,'dt':dt,'seed':seed,'samples':len(X),
          'splits':{name:{'trajectories':int(np.sum(group==i)),'windows':int(np.sum(split==i))}
              for i,name in enumerate(('train','validation','calibration','test'))},
          'boundary_windows_excluded':boundary_rejected,'generation_seconds':time.perf_counter()-tic,
          'source':'synthetic PI trajectories, P25 Table2 limits; no real sonar data',
          'split_rule':'whole independent trajectory; training normalization uses split 0 only',
          'reconstruction_choices':['PI anti-windup','boundary retreat','dt','seed','discrete integration']}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2));return meta

def load_dataset(path):
    with np.load(path,allow_pickle=False) as d: data={k:d[k] for k in d.files}
    for i in range(4):
        if not np.any(data['split']==i):raise ValueError('empty data split')
    groups=[set(data['trajectory_id'][data['split']==i]) for i in range(4)]
    if any(groups[i]&groups[j] for i in range(4) for j in range(i+1,4)):
        raise ValueError('trajectory leakage between splits')
    return data
