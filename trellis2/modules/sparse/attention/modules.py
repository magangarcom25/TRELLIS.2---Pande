from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import VarLenTensor, SparseTensor
from ... import sparse as sp  
from .full_attn import sparse_scaled_dot_product_attention
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention
from .rope import SparseRotaryPositionEmbedder

class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: Union[VarLenTensor, torch.Tensor]) -> Union[VarLenTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, VarLenTensor):
            x = x.replace(F.normalize(x.feats, dim=-1) * self.gamma * self.scale)
        else:
            x = F.normalize(x, dim=-1) * self.gamma * self.scale
        return x.to(x_type)

class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed", "double_windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        q_bias: bool = True,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.head_dim = channels // num_heads
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        
        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = SparseRotaryPositionEmbedder(self.head_dim, rope_freq=rope_freq)

    @staticmethod
    def _linear(module: nn.Linear, x: Union[VarLenTensor, torch.Tensor, SparseTensor]) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, (VarLenTensor, SparseTensor)):
            return x.replace(module(x.feats))
        return module(x)

    def _reshape_chs(self, x: Union[VarLenTensor, torch.Tensor, SparseTensor], num_heads: int) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, (VarLenTensor, SparseTensor)):
            new_shape = (num_heads, x.feats.shape[-1] // num_heads)
            return x.reshape(*new_shape)
        else:
            B, L, C = x.shape
            D = C // num_heads
            return x.reshape(B, L, num_heads, D)

    def _fused_pre(self, x: Union[VarLenTensor, torch.Tensor], num_fused: int) -> Union[VarLenTensor, torch.Tensor]:
        is_varlen = isinstance(x, VarLenTensor)
        feats = x.feats if is_varlen else x
        total_dim = feats.shape[-1]
        D = self.head_dim
        H = (total_dim // num_fused) // D
        
        if feats.dim() == 2:
            new_feats = feats.reshape(-1, num_fused, H, D)
            return x.replace(new_feats) if is_varlen else new_feats
        elif feats.dim() == 3:
            B, L, _ = feats.shape
            new_feats = feats.reshape(B, L, num_fused, H, D)
            return new_feats
        return x

    def forward(self, x: SparseTensor, context: Optional[Union[VarLenTensor, torch.Tensor]] = None) -> SparseTensor:
        if self._type == "self":
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            
            if self.qk_rms_norm or self.use_rope:
                q, k, v = qkv.unbind(dim=-3)
                if self.qk_rms_norm:
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                if self.use_rope:
                    q, k = self.rope(q, k)
                
                combined = torch.stack([q.feats, k.feats, v.feats], dim=-3)
                qkv = qkv.replace(combined)

            if self.attn_mode == "full":
                h = sparse_scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv, self.window_size, shift_window=self.shift_window
                )
            else: # double_windowed
                qkv0 = qkv.replace(qkv.feats[:, :self.num_heads//2])
                qkv1 = qkv.replace(qkv.feats[:, self.num_heads//2:])
                h0 = sparse_windowed_scaled_dot_product_self_attention(qkv0, self.window_size)
                h1 = sparse_windowed_scaled_dot_product_self_attention(qkv1, self.window_size, shift_window=tuple([self.window_size//2]*3))
                h = qkv.replace(torch.cat([h0.feats, h1.feats], dim=1))
        else:
            # --- CROSS ATTENTION EMERGENCY SYNC ---
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, self.num_heads)
            kv_raw = self._linear(self.to_kv, context)
            
            target_batch = kv_raw.shape[0] if not isinstance(kv_raw, VarLenTensor) else kv_raw.shape[0]

            # FORCE SYNC Q
            if q.shape[0] == 1 and target_batch == 2:
                q = sp.sparse_cat([q, q])
            
            if not isinstance(kv_raw, VarLenTensor):
                if kv_raw.dim() == 3:
                    B, L, C = kv_raw.shape
                else:
                    B, C = target_batch, kv_raw.shape[-1]
                    L = kv_raw.shape[0] // B
                feats_kv = kv_raw.reshape(-1, C)
                kv_off = torch.arange(0, B + 1, device=feats_kv.device) * L
                kv = VarLenTensor(feats_kv, kv_off.long())
            else:
                kv = kv_raw

            kv = self._fused_pre(kv, num_fused=2)
            k, v = kv.unbind(dim=-3)
            
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)

            h = sparse_scaled_dot_product_attention(q, k, v)
                
        if isinstance(h, VarLenTensor):
            h = h.replace(h.feats.reshape(h.feats.shape[0], -1))
        else:
            h = h.reshape(*h.shape[:-2], -1)

        h = self._linear(self.to_out, h)
        return h