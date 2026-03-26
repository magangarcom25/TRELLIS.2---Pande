from typing import *
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to, manual_cast, str_to_dtype
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
from .sparse_structure_flow import TimestepEmbedder
from .sparse_elastic_mixin import SparseTransformerElasticMixin

class SLatFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = 'bfloat16',
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = 'vanilla',
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        
        self.dtype = torch.bfloat16 

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels)
            
        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                rope_freq=rope_freq,
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])
            
        self.out_layer = sp.SparseLinear(model_channels, out_channels)

        self.initialize_weights()
        # Inisialisasi awal ke BF16
        self.to(torch.bfloat16)
        
        # --- KHUSUS V6.8: Kembalikan t_embedder ke Float32 agar stabil ---
        self.t_embedder.to(torch.float32)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.LayerNorm, sp.SparseLinear)):
                module.to(dtype)
        self.blocks.apply(partial(convert_module_to, dtype=dtype))
        # Pastikan t_embedder tetap di float32
        self.t_embedder.to(torch.float32)

    def initialize_weights(self) -> None:
        if self.initialization == 'vanilla':
            def _basic_init(module):
                if isinstance(module, (nn.Linear, sp.SparseLinear)):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if hasattr(module, 'bias') and module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor], Dict],
        concat_cond: Optional[sp.SparseTensor] = None,
        **kwargs
    ) -> sp.SparseTensor:
        # --- SURGERY V6.8: HYBRID BF16/FP32 PROTOCOL ---
        target_dtype = torch.bfloat16
        
        # 1. Pastikan model utama tetap di BF16
        if self.input_layer.weight.dtype != target_dtype:
            self.to(target_dtype)
            self.t_embedder.to(torch.float32) # Selalu paksa t_embedder kembali ke FP32

        # 2. Paksa input Tensor ke BF16
        x = x.replace(x.feats.to(target_dtype))
        
        # 3. Conditioning Cleanup
        if isinstance(cond, list):
            cond = [c.to(target_dtype) for c in cond]
            cond = sp.VarLenTensor.from_tensor_list(cond)
        elif isinstance(cond, dict):
            cond = {k: v.to(target_dtype) if isinstance(v, torch.Tensor) else v for k, v in cond.items()}
        elif isinstance(cond, torch.Tensor):
            cond = cond.to(target_dtype)

        # 4. Input Layer
        h = self.input_layer(x)
        
        # 5. Timestep Embedding (DUNIA FLOAT32)
        # Kita hitung di Float32, lalu paksa hasilnya ke BF16
        t_emb = self.t_embedder(t.float()).to(target_dtype)
        
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
            t_emb = t_emb.to(target_dtype)

        # 6. Position Embedding
        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + pe.to(target_dtype)
            
        # 7. Transformer Blocks (DUNIA BFLOAT16 + FlashAttention)
        for block in self.blocks:
            h = block(h, t_emb, cond)

        # 8. Final Normalization & Output
        h_feats = h.feats.to(target_dtype)
        h_feats = F.layer_norm(h_feats, h_feats.shape[-1:])
        h = h.replace(h_feats)
        
        h = self.out_layer(h)
            
        return h

class ElasticSLatFlowModel(SparseTransformerElasticMixin, SLatFlowModel):
    pass