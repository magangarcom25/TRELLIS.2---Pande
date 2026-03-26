from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor

class SparseRotaryPositionEmbedder(nn.Module):
    def __init__(
        self, 
        head_dim: int,
        dim: int = 3,
        rope_freq: Tuple[float, float] = (1.0, 10000.0)
    ):
        super().__init__()
        assert head_dim % 2 == 0, "Head dim must be divisible by 2"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        
        # Inisialisasi frekuensi posisi
        freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        freqs = rope_freq[0] / (rope_freq[1] ** (freqs))
        self.register_buffer("freqs", freqs)
        
    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        """Menghitung fase rotasi dalam Float32 untuk stabilitas."""
        indices = indices.to(torch.float32)
        phases_val = torch.outer(indices, self.freqs)
        
        # torch.polar memerlukan Float32/Float64 (tidak support BFloat16/Half)
        ones = torch.ones_like(phases_val)
        phases = torch.polar(ones, phases_val)
        return phases
        
    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """
        Melakukan rotasi dengan penanganan otomatis untuk kelipatan batch (CFG/Multi-view).
        """
        # --- LOGIKA DYNAMIC MULTIPLIER PANDE (FIX 3048 vs 762) ---
        if x.shape[0] != phases.shape[0]:
            multiplier = x.shape[0] // phases.shape[0]
            # Ulangi phases sebanyak kelipatan yang dibutuhkan (misal 4x)
            phases = torch.cat([phases] * multiplier, dim=0)
        # -------------------------------------------------------

        # Simpan dtype asli (misal BFloat16)
        original_dtype = x.dtype
        
        # Lakukan komputasi dalam Float32 & Complex64
        x_float = x.to(torch.float32)
        phases_complex = phases.to(torch.complex64)
        
        # Reshape fitur ke bilangan kompleks: [N, H, D/2, 2] -> [N, H, D/2]
        # Kita gunakan reshape manual agar lebih aman terhadap dimensi 'H' (heads)
        x_complex = torch.view_as_complex(x_float.reshape(*x.shape[:-1], -1, 2))
        
        # Lakukan rotasi (perkalian kompleks)
        # unsqueeze(-2) memastikan phase meluas ke seluruh Head (H)
        x_rotated = x_complex * phases_complex.unsqueeze(-2)
        
        # Kembalikan ke Real [N, H, D] dan kembalikan ke dtype awal
        x_embed = torch.view_as_real(x_rotated).reshape(*x.shape).to(original_dtype)
        return x_embed
        
    def forward(self, q: SparseTensor, k: Optional[SparseTensor] = None) -> Union[SparseTensor, Tuple[SparseTensor, SparseTensor]]:
        """
        Forward pass untuk Query dan Key.
        """
        assert q.coords.shape[-1] == self.dim + 1, f"Coords dim mismatch. Expected {self.dim + 1}"
        
        phases_cache_name = f'rope_phase_{self.dim}d_freq{self.rope_freq[0]}-{self.rope_freq[1]}_hd{self.head_dim}'
        phases = q.get_spatial_cache(phases_cache_name)
        
        if phases is None:
            # Ambil koordinat XYZ (abaikan batch index di kolom 0)
            coords = q.coords[..., 1:] 
            
            # Hitung phases awal
            phases = self._get_phases(coords.reshape(-1)).reshape(*coords.shape[:-1], -1)
            
            # Padding jika head_dim tidak habis dibagi rata oleh dimensi spasial
            if phases.shape[-1] < self.head_dim // 2:
                padn = self.head_dim // 2 - phases.shape[-1]
                padding = torch.polar(
                    torch.ones(*phases.shape[:-1], padn, device=phases.device, dtype=torch.float32),
                    torch.zeros(*phases.shape[:-1], padn, device=phases.device, dtype=torch.float32)
                )
                phases = torch.cat([phases, padding], dim=-1)
                
            # Simpan ke cache agar tidak hitung ulang di block berikutnya
            q.register_spatial_cache(phases_cache_name, phases)
        
        # Apply RoPE ke Query
        q_embed = q.replace(self._rotary_embedding(q.feats, phases))
        
        if k is None:
            return q_embed
            
        # Apply RoPE ke Key
        k_embed = k.replace(self._rotary_embedding(k.feats, phases))
        
        return q_embed, k_embed