from typing import *
import torch
from .. import VarLenTensor
from .. import config

__all__ = [
    'sparse_scaled_dot_product_attention',
]

def get_seqlen(x, batch_size):
    """Fungsi detektif untuk menghitung seqlen tanpa crash."""
    # Skenario 1: Objek adalah VarLenTensor yang punya layout lengkap
    if hasattr(x, 'layout') and x.layout is not None:
        try:
            return [x.layout[i].stop - x.layout[i].start for i in range(batch_size)]
        except (AttributeError, IndexError):
            pass
            
    # Skenario 2: Objek adalah Tensor polos (hasil split/cat)
    # Jika x adalah [B, L, H, C]
    if x.dim() >= 3 and x.shape[0] == batch_size:
        return [x.shape[1]] * batch_size
    
    # Skenario 3: Objek adalah Flattened Features [Total_Len, H, C]
    total_len = x.shape[0]
    return [total_len // batch_size] * batch_size

def sparse_scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {1: ['qkv'], 2: ['q', 'kv'], 3: ['q', 'k', 'v']}
    num_all_args = len(args) + len(kwargs)
    
    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        device = qkv.device
        s = qkv
        q_seqlen = get_seqlen(qkv, qkv.shape[0])
        kv_seqlen = q_seqlen
        qkv_feats = qkv.feats if hasattr(qkv, 'feats') else qkv

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        
        # Emergency duplication untuk CFG
        if q.shape[0] == 1 and kv.shape[0] == 2:
            import trellis2.modules.sparse as sp_root
            q = sp_root.sparse_cat([q, q])

        device = q.device
        q_seqlen = get_seqlen(q, q.shape[0])
        kv_seqlen = get_seqlen(kv, kv.shape[0])
        s = q if isinstance(q, VarLenTensor) else None
        q_feats = q.feats if hasattr(q, 'feats') else q
        kv_feats = kv.feats if hasattr(kv, 'feats') else kv

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        
        # Emergency duplication untuk CFG
        if q.shape[0] == 1 and k.shape[0] == 2:
            import trellis2.modules.sparse as sp_root
            q = sp_root.sparse_cat([q, q])

        device = q.device
        q_seqlen = get_seqlen(q, q.shape[0])
        kv_seqlen = get_seqlen(k, k.shape[0])
        s = q if isinstance(q, VarLenTensor) else None
        
        q_feats = q.feats if hasattr(q, 'feats') else q
        k_feats = k.feats if hasattr(k, 'feats') else k
        v_feats = v.feats if hasattr(v, 'feats') else v

    # --- Eksekusi Attention Backend ---
    if config.ATTN == 'flash_attn':
        if 'flash_attn' not in globals(): import flash_attn
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
        
        if num_all_args == 1:
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, cu_seqlens_q, max(q_seqlen))
        elif num_all_args == 2:
            out = flash_attn.flash_attn_varlen_kvpacked_func(q_feats, kv_feats, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
        elif num_all_args == 3:
            out = flash_attn.flash_attn_varlen_func(q_feats, k_feats, v_feats, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
    else:
        raise ValueError(f"Backend {config.ATTN} tidak didukung dalam mode robust Ganesha.")
    
    if s is not None and hasattr(s, 'replace'):
        return s.replace(out)
    return out