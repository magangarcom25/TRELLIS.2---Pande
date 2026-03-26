import torch
import torch.nn as nn
from typing import *
from trellis2.models.base import BaseModel
from trellis2.modules import sparse as sp

class SparseResBlock(nn.Module):
    def __init__(self, channels, emb_channels):
        super().__init__()
        self.gn1 = sp.SparseGroupNorm(32, channels)
        self.act = nn.SiLU() 
        self.conv1 = sp.SparseConv3d(channels, channels, kernel_size=3, padding=None)
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, channels)
        )
        
        self.gn2 = sp.SparseGroupNorm(32, channels)
        self.conv2 = sp.SparseConv3d(channels, channels, kernel_size=3, padding=None)

    def forward(self, x: sp.SparseTensor, emb: torch.Tensor) -> sp.SparseTensor:
        target_dtype = x.feats.dtype
        target_device = x.feats.device

        # Block 1
        h_norm = self.gn1(x)
        h_feats = self.act(h_norm.feats)
        h = sp.SparseTensor(feats=h_feats, coords=x.coords, shape=x.shape)
        h = self.conv1(h)
        
        # Timestep Embedding
        emb_out = self.emb_layers(emb.to(device=target_device, dtype=target_dtype))
        h = sp.SparseTensor(
            feats=h.feats + emb_out[h.coords[:, 0]], 
            coords=h.coords, 
            shape=h.shape
        )
        
        # Block 2
        h_norm2 = self.gn2(h)
        h_feats2 = self.act(h_norm2.feats)
        h = sp.SparseTensor(feats=h_feats2, coords=h.coords, shape=h.shape)
        h = self.conv2(h)
        
        return sp.SparseTensor(feats=x.feats + h.feats, coords=x.coords, shape=x.shape)


class SparseStructureFlowModel(BaseModel):
    def __init__(self, resolution, in_channels, model_channels, out_channels, num_res_blocks=12, **kwargs):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels 
        self.out_channels = out_channels
        
        self.time_in = nn.Linear(1, model_channels) 
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels),
        )
        
        self.input_layer = sp.SparseConv3d(in_channels, model_channels, kernel_size=3, padding=None)
        self.res_blocks = nn.ModuleList([
            SparseResBlock(model_channels, model_channels) for _ in range(num_res_blocks)
        ])
        
        self.gn_out = sp.SparseGroupNorm(32, model_channels)
        self.act_out = nn.SiLU()
        self.out_layer = sp.SparseConv3d(model_channels, out_channels, kernel_size=3, padding=None)

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: dict = None) -> sp.SparseTensor:
        param = next(self.parameters())
        master_device = param.device
        master_dtype = param.dtype

        if x.feats.shape[-1] != self.in_channels:
            feats = x.feats[:, :self.in_channels].contiguous().to(device=master_device, dtype=master_dtype)
        else:
            feats = x.feats.to(device=master_device, dtype=master_dtype)

        x_sparse = sp.SparseTensor(feats=feats, coords=x.coords.to(master_device), shape=x.shape)

        if t.ndim == 1: t = t.unsqueeze(-1)
        t_emb = self.time_embed(self.time_in(t.to(device=master_device, dtype=master_dtype)))

        h = self.input_layer(x_sparse)
        for block in self.res_blocks:
            h = block(h, t_emb)
            
        h = self.gn_out(h)
        h_final_feats = self.act_out(h.feats)
        h = sp.SparseTensor(feats=h_final_feats, coords=h.coords, shape=h.shape)
        
        return self.out_layer(h)