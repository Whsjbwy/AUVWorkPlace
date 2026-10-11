from __future__ import annotations
from pathlib import Path
import json, math, time
import numpy as np
import torch
from .learning import PPLNM,training_loss
from .data import load_dataset
from .prediction import SphericalCalibration,pplnm_contract


def dataset_contract(data,path,xmean,xstd,yscale):
    """Inspectable training fields, not a proof of byte-identical datasets."""
    X,Y=data['X'],data['Y']
    if X.ndim!=3 or X.shape[1:]!=(6,9) or Y.shape!=(len(X),3,3) or data['physical'].shape!=Y.shape:
        raise ValueError('dataset requires X [N,6,9], Y/physical [N,3,3]')
    if any(not np.isfinite(data[key]).all() for key in ('X','Y','physical','speed')):
        raise ValueError('dataset contains nonfinite learning values')
    dt=float(data['dt'])
    if not np.isfinite(dt) or dt<=0:raise ValueError('dataset dt must be finite and positive')
    contract={
        'sample_count':len(X),'input_dim':9,'output_dim':3,'history_length':6,
        'horizon_steps':3,'prediction_dt_s':dt,
        'array_shapes':{key:list(data[key].shape) for key in ('X','Y','physical','split','trajectory_id','speed')},
        'array_dtypes':{key:str(data[key].dtype) for key in ('X','Y','physical','split','trajectory_id','speed')},
        'split_sample_counts':[int(np.sum(data['split']==i)) for i in range(4)],
        'split_trajectory_ids':[np.unique(data['trajectory_id'][data['split']==i]).tolist() for i in range(4)],
        'training_xmean':xmean.tolist(),'training_xstd':xstd.tolist(),'training_yscale':yscale.tolist(),
    }
    sidecar=Path(path).with_suffix('.json')
    if sidecar.is_file():
        metadata=json.loads(sidecar.read_text(encoding='utf-8'))
        contract['generation_config']={key:metadata[key] for key in
            ('n_trajectories','seconds','dt','seed','samples','source','split_rule','reconstruction_choices') if key in metadata}
    return contract

def train_predictor(data_path,out,epochs=12,seed=0,device='cpu',batch_size=256,threads=2,overwrite=False,chunk_epochs=None,resume=False):
    out=Path(out)
    if out.exists() and any(out.iterdir()) and not overwrite and not resume:raise FileExistsError(out)
    if not isinstance(epochs,int) or epochs<1:raise ValueError('epochs must be positive')
    if not isinstance(threads,int) or threads<1:raise ValueError('threads must be positive')
    if not isinstance(batch_size,int) or batch_size<1:raise ValueError('batch_size must be positive')
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(threads);torch.manual_seed(seed)
    np.random.seed(seed);d=load_dataset(data_path);tr=d['split']==0;va=d['split']==1;dt=float(d['dt'])
    xmean=d['X'][tr].mean((0,1));xstd=np.maximum(d['X'][tr].std((0,1)),1e-5)
    residual=d['Y']-d['physical'];yscale=np.maximum(residual[tr].std((0,1)),.05)
    X=(d['X']-xmean)/xstd;R=residual/yscale
    model=PPLNM().to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=5e-4)
    train_idx=np.flatnonzero(tr);rng=np.random.default_rng(seed);history=[];best=float('inf');best_state=None
    tic=time.perf_counter();start_epoch=0;previous_wall_s=0.
    data_config=dataset_contract(d,data_path,xmean,xstd,yscale)
    training_config={'total_epochs':epochs,'seed':seed,'batch_size':batch_size,
                     'model':pplnm_contract(dt,epochs,'training_in_progress')}
    if resume:
        state=torch.load(out/'training_state.pt',map_location='cpu',weights_only=True)
        if state.get('dataset_config')!=data_config or state.get('training_config')!=training_config:
            raise ValueError('resume requires the same dataset configuration/normalization, total epochs, seed and batch size; legacy states must be restarted')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);rng.bit_generator.state=json.loads(state['numpy_rng'])
        start_epoch=state['completed'];history=state['history'];best=state['best'];best_state=state['best_state']
        previous_wall_s=state['wall_clock_s']
    stop_epoch=epochs if chunk_epochs is None else min(epochs,start_epoch+chunk_epochs)
    if chunk_epochs is not None and chunk_epochs<1:raise ValueError('chunk_epochs must be positive')
    def save_training_state(completed):
        state={'model':{k:z.detach().cpu().clone() for k,z in model.state_dict().items()},
               'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),
               'numpy_rng':json.dumps(rng.bit_generator.state),'completed':completed,
               'total_epochs':epochs,'seed':seed,'history':history,'best':best,'best_state':best_state,
               'wall_clock_s':previous_wall_s+time.perf_counter()-tic,
               'dataset_config':data_config,'training_config':training_config}
        temp=out/'training_state.tmp';torch.save(state,temp);temp.replace(out/'training_state.pt')
    def evaluate(mask):
        model.eval();pred=[]
        with torch.inference_mode():
            idx=np.flatnonzero(mask)
            for start in range(0,len(idx),batch_size):
                j=idx[start:start+batch_size]
                r=model(torch.as_tensor(X[j],device=device)).cpu().numpy()*yscale
                pred.append(d['physical'][j]+r)
        return np.concatenate(pred)
    for epoch in range(start_epoch,stop_epoch):
        # Paper's 20-epoch warmup retained, even for a short engineering smoke run.
        scale=(epoch+1)/20 if epoch<20 else .5*(1+math.cos(math.pi*(epoch-20)/max(1,epochs-20)))
        for group in opt.param_groups:group['lr']=1e-4*scale
        model.train();losses=[]
        for j in np.array_split(rng.permutation(train_idx),max(1,int(np.ceil(len(train_idx)/batch_size)))):
            tensors=[torch.as_tensor(a[j],device=device) for a in (X,R,d['physical'],d['Y'])]
            x,r,p,y=tensors;opt.zero_grad(set_to_none=True)
            loss=training_loss(model(x),r,p,y,torch.as_tensor(yscale,device=device),dt)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach().cpu()))
        val=evaluate(va);v=float(np.mean((val-d['Y'][va])**2))
        if not np.isfinite(v):raise FloatingPointError('nonfinite validation prediction/loss')
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation_mse_m2':v,'lr':opt.param_groups[0]['lr']})
        if v<best:best=v;best_state={k:z.detach().cpu().clone() for k,z in model.state_dict().items()}
        if (epoch+1)%5==0 or epoch+1==stop_epoch:
            save_training_state(epoch+1)
            print(f'epoch {epoch+1}/{epochs}, validation MSE {v:.6f}',flush=True)
    if stop_epoch<epochs:
        progress={'status':'training_chunk_complete','epochs_completed':stop_epoch,'total_epochs':epochs,
                  'best_validation_mse_m2':best,'wall_clock_s':previous_wall_s+time.perf_counter()-tic,
                  'note':'No test-set evaluation until all epochs complete. Resume with --resume.'}
        (out/'training_progress.json').write_text(json.dumps(progress,indent=2))
        return progress
    model.load_state_dict(best_state)
    metrics={}
    for split,name in ((1,'validation'),(2,'calibration'),(3,'test')):
        mask=d['split']==split;pred=evaluate(mask);truth=d['Y'][mask]
        metrics[name]={'n':int(mask.sum()),'pplnm_rmse_coordinate_m':float(np.sqrt(np.mean((pred-truth)**2))),
           'ca_rmse_coordinate_m':float(np.sqrt(np.mean((d['physical'][mask]-truth)**2))),
           'pplnm_euclidean_error_by_horizon_m':np.linalg.norm(pred-truth,axis=-1).mean(0).tolist()}
        if split==2:
            cal=SphericalCalibration().fit(pred,truth,d['speed'][mask]);cal.save(out/'pplnm_calibration.json')
            ca_cal=SphericalCalibration().fit(d['physical'][mask],truth,d['speed'][mask]);ca_cal.save(out/'ca_calibration.json')
        if split==3:
            qs=np.array([cal.radius(s,np.arange(1,4)*dt,dt) for s in d['speed'][mask]])
            covered=np.linalg.norm(pred-truth,axis=-1)<=qs
            metrics[name]['marginal_coverage']=covered.mean(0).tolist();metrics[name]['three_step_joint_coverage']=float(covered.all(1).mean())
    checkpoint={'model':best_state,'xmean':xmean.tolist(),'xstd':xstd.tolist(),'yscale':yscale.tolist(),
                'dt':dt,'epochs_completed':epochs,'training_label':'engineering_smoke_not_full_150_epoch_reproduction' if epochs<150 else '150_epochs_small_synthetic_dataset_not_full_paper_reproduction',
                'seed':seed,'heads':'3 independent Linear(256,3); paper head internals unspecified'}
    checkpoint['model_config']=pplnm_contract(dt,epochs,checkpoint['training_label'])
    torch.save(checkpoint,out/'pplnm.pt')
    cal.metadata=checkpoint['model_config'].copy()
    ca_cal.metadata={'predictor_name':'constant_acceleration_debug_baseline','prediction_dt_s':dt,
                     'input_dim':9,'output_dim':3,'history_length':6,'horizon_steps':3}
    cal.save(out/'pplnm_calibration.json');ca_cal.save(out/'ca_calibration.json')
    # Re-open checkpoint: serialization is part of the actual smoke test.
    reopened=torch.load(out/'pplnm.pt',weights_only=True,map_location='cpu');copy=PPLNM();copy.load_state_dict(reopened['model']);copy.eval();model.eval()
    z=torch.tensor(X[:4]);
    with torch.inference_mode(): reload_error=float(torch.max(torch.abs(copy(z)-model(z.to(device)).cpu())))
    report={'epochs_completed':epochs,'best_validation_mse_m2':best,'parameter_count':sum(p.numel() for p in model.parameters()),
            'device':device,'torch':torch.__version__,'seed':seed,'wall_clock_s':previous_wall_s+time.perf_counter()-tic,
            'checkpoint_reload_max_abs_error':reload_error,'metrics':metrics,'claim':checkpoint['training_label'],
            'dataset_config':data_config,
            'compatibility_note':'Actual model/data parameters are checked; this does not prove identical dataset contents.'}
    report['best_epoch_by_validation']=min(history,key=lambda row:row['validation_mse_m2'])['epoch']
    (out/'training_progress.json').write_text(json.dumps({'status':'complete','epochs_completed':epochs,'total_epochs':epochs},indent=2))
    (out/'training_report.json').write_text(json.dumps(report,indent=2));(out/'training_history.json').write_text(json.dumps(history,indent=2))
    return report
