import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import manual_cast

class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        # Paksa input ke float32
        x = x.float()
        
        # Bypass super().forward dan panggil functional secara manual
        # Kita pastikan weight dan bias juga ikut jadi float32 saat perhitungan
        o = F.layer_norm(
            x, 
            self.normalized_shape, 
            self.weight.float() if self.weight is not None else None, 
            self.bias.float() if self.bias is not None else None, 
            self.eps
        )
        return o.to(x_dtype)

class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        
        # Bypass super().forward untuk GroupNorm
        o = F.group_norm(
            x,
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps
        )
        return o.to(x_dtype)

class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        # Permute ke (N, ..., C)
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        # Balikkan ke (N, C, ...)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x