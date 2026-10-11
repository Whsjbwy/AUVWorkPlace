from __future__ import annotations
import numpy as np
import torch
from torch import nn

class PPLNM(nn.Module):
    """P25 Tables 3/5 structure; head internals were not specified.

    Two 256-unit LSTM layers, dropout 0.2, LayerNorm, 3 independent linear
    regression heads. Linear-head choice is disclosed, not asserted exact.
    """
    def __init__(self):
        super().__init__();self.lstm=nn.LSTM(9,256,num_layers=2,dropout=.2,batch_first=True)
        self.norm=nn.LayerNorm(256);self.heads=nn.ModuleList([nn.Linear(256,3) for _ in range(3)])
        # Reconstruction choice: begin at the CA baseline, not a random residual.
        for head in self.heads:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(6,9):raise ValueError('expected [batch,6,9]')
        h,_=self.lstm(x);z=self.norm(h[:,-1]);return torch.stack([head(z) for head in self.heads],1)

def training_loss(pred_residual, target_residual, physical, truth, yscale,dt):
    mse=nn.functional.mse_loss(pred_residual,target_residual)
    predicted=physical+pred_residual*yscale
    v_pred=torch.diff(predicted,dim=1)/dt;v_true=torch.diff(truth,dim=1)/dt
    return mse+.4*nn.functional.smooth_l1_loss(v_pred,v_true)
