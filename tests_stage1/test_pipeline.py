from dataclasses import replace
import json
import numpy as np
import pytest
from auv3d.config import ExperimentConfig
from auv3d.simulation import run_episode,save_episode
from auv3d.data import generate_dataset,load_dataset

def test_empty_episode_moves_in_all_three_dimensions():
    r=run_episode(ExperimentConfig(),'empty3d')
    assert r['summary']['status']=='success';assert np.all(np.abs(r['states'][-1,1:4]-r['states'][0,1:4])>10)
    assert r['summary']['energy_joules'] is None

def test_episode_same_seed_same_states():
    c=ExperimentConfig();c=replace(c,planner=replace(c.planner,max_time_s=8))
    a=run_episode(c,'crossing3d');b=run_episode(c,'crossing3d');np.testing.assert_array_equal(a['states'],b['states'])

def test_output_overwrite_requires_explicit_flag(tmp_path):
    c=ExperimentConfig();c=replace(c,planner=replace(c.planner,max_time_s=1))
    r=run_episode(c);save_episode(r,tmp_path/'test',plot=False)
    with pytest.raises(FileExistsError):save_episode(r,tmp_path/'test',plot=False)
    save_episode(r,tmp_path/'test',overwrite=True,plot=False)

def test_whole_trajectory_splits(tmp_path):
    p=tmp_path/'tiny.npz';meta=generate_dataset(p,n_trajectories=20,seconds=18,dt=1.)
    d=load_dataset(p);sets=[set(d['trajectory_id'][d['split']==i]) for i in range(4)]
    assert all(not(sets[i]&sets[j]) for i in range(4) for j in range(i+1,4))
    assert d['X'].shape[1:]==(6,9);assert d['Y'].shape[1:]==(3,3)
    assert sum(x['windows'] for x in meta['splits'].values())==len(d['X'])

def test_lstm_forward_backward_and_reload(tmp_path):
    torch=pytest.importorskip('torch');torch.set_num_threads(2)
    from auv3d.learning import PPLNM,training_loss
    model=PPLNM();x=torch.randn(4,6,9);r=model(x);assert r.shape==(4,3,3)
    loss=(r-torch.ones_like(r)).square().mean();loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    p=tmp_path/'m.pt';torch.save(model.state_dict(),p);new=PPLNM();new.load_state_dict(torch.load(p,weights_only=True))
    model.eval();new.eval();assert torch.equal(model(x),new(x))

def test_network_rejects_wrong_feature_count():
    torch=pytest.importorskip('torch');from auv3d.learning import PPLNM
    with pytest.raises(ValueError):PPLNM()(torch.zeros(2,6,10))
