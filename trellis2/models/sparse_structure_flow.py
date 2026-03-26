from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))

class SparseOutput:
    def __init__(self, feats):
        self.feats = feats

class SparseStructureFlowModel(nn.Module):
    def __init__(self, resolution, in_channels, model_channels, cond_channels, out_channels, num_blocks, **kwargs):
        super().__init__()
        self.resolution = resolution
        self.out_channels = out_channels
        
        # ADAPTERS: Sinkronisasi 32-ch Dataset <-> 8-ch Checkpoint
        self.input_adapter = nn.Linear(in_channels, 8)
        self.output_adapter = nn.Linear(8, out_channels)

        self.t_embedder = TimestepEmbedder(model_channels)
        self.input_layer = nn.Linear(8, 128) 
        self.proj_in = nn.Linear(128, model_channels)
        self.cond_proj = nn.Linear(cond_channels, model_channels)
        
        pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
        coords = torch.meshgrid(*[torch.arange(res) for res in [resolution] * 3], indexing='ij')
        coords = torch.stack(coords, dim=-1).reshape(-1, 3).float()
        self.register_buffer("pos_emb", pos_embedder(coords))

        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(model_channels, model_channels, num_heads=kwargs.get('num_heads', 16), mlp_ratio=4, use_checkpoint=True)
            for _ in range(num_blocks)
        ])

        self.proj_out = nn.Linear(model_channels, 128)
        self.out_layer = nn.Linear(128, 8)

    def forward(self, x: Any, t: torch.Tensor, cond: torch.Tensor) -> Any:
        if isinstance(x, dict):
            x_data = x.get('feats', x.get('features', next(iter(x.values()))))
            coords = x.get('coords', None)
        else:
            x_data = getattr(x, 'features', getattr(x, 'feats', x))
            coords = getattr(x, 'indices', getattr(x, 'coords', None))

        B, R = t.shape[0], self.resolution
        ref_dtype = self.proj_in.weight.dtype

        with torch.amp.autocast('cuda', enabled=True, dtype=ref_dtype):
            if coords is not None:
                actual_feats = x_data.features if hasattr(x_data, 'features') else x_data
                feats_8 = self.input_adapter(actual_feats.to(ref_dtype))
                h_dense = torch.zeros((B, 8, R, R, R), device=t.device, dtype=ref_dtype)
                c = coords.long()
                h_dense[c[:, 0], :, c[:, 1], c[:, 2], c[:, 3]] = feats_8
                h = h_dense.flatten(2).permute(0, 2, 1).contiguous()
            else:
                h = self.input_adapter(x_data.to(ref_dtype)).flatten(2).permute(0, 2, 1).contiguous()

            h = self.proj_in(self.input_layer(h))
            h_cond = self.cond_proj(cond.to(ref_dtype))
            t_emb = self.t_embedder(t).to(ref_dtype)
            
            for block in self.blocks:
                h = block(h + self.pos_emb[None].to(h.dtype), t_emb, h_cond)

            h = self.out_layer(self.proj_out(F.layer_norm(h, h.shape[-1:])))

        h_32 = self.output_adapter(h)
        out_dense = h_32.permute(0, 2, 1).view(B, self.out_channels, R, R, R)
        
        if coords is not None:
            c = coords.long()
            out_sparse = out_dense[c[:, 0], :, c[:, 1], c[:, 2], c[:, 3]]
            return SparseOutput(out_sparse.to(ref_dtype))
        
        return SparseOutput(out_dense.to(ref_dtype))