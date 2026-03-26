import importlib
import os
import json
from typing import Any

# 1. Mapping Class ke Modul (Gunakan path relatif dari folder models)
__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae',

    # Feature Extractors (Handling khusus di __getattr__)
    'DinoV2FeatureExtractor': 'trainers.flow_matching.mixins.image_conditioned',
    'DinoV3FeatureExtractor': 'trainers.flow_matching.mixins.image_conditioned',
}

__submodules = []
__all__ = list(__attributes.keys()) + __submodules + ['from_pretrained']

def __getattr__(name):
    """Lazy loading untuk menghemat VRAM dan mempercepat startup"""
    if name in __attributes:
        module_path = __attributes[name]
        
        # Logika Penentuan Path Import
        if module_path.startswith('trainers.'):
            # Import dari root trellis2
            full_path = f"trellis2.{module_path}"
        else:
            # Import relatif dari folder models saat ini
            full_path = f"{__name__}.{module_path}"
        
        try:
            module = importlib.import_module(full_path)
            cls = getattr(module, name)
            globals()[name] = cls  # Cache ke globals agar pemanggilan kedua lebih cepat
            return cls
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Gagal memuat {name} dari {full_path}. Error: {e}")
            
    if name in __submodules:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
        
    raise AttributeError(f"Modul {__name__} tidak memiliki atribut '{name}'")


def from_pretrained(path: str, torch_dtype=None, device="cpu", **kwargs):
    """
    Load model dari lokal (Ganesha 600M) atau HuggingFace Hub.
    """
    from safetensors.torch import load_file
    import torch

    # 1. Resolusi Path (Hapus ekstensi jika user iseng memasukkannya)
    base_path = os.path.splitext(path)[0]
    config_file = f"{base_path}.json"
    model_file = f"{base_path}.safetensors"

    # 2. Cek apakah file ada di lokal
    if os.path.exists(config_file) and os.path.exists(model_file):
        print(f"[*] Loading model dari lokal: {base_path}")
    else:
        # Jika tidak ada, coba cari di HuggingFace
        try:
            from huggingface_hub import hf_hub_download
            print(f"[*] Mencari di HuggingFace Hub: {path}")
            
            # Format path HF: 'author/repo/model_name'
            parts = path.split('/')
            repo_id = f"{parts[0]}/{parts[1]}"
            filename_prefix = "/".join(parts[2:])
            
            config_file = hf_hub_download(repo_id, f"{filename_prefix}.json")
            model_file = hf_hub_download(repo_id, f"{filename_prefix}.safetensors")
        except Exception as e:
            raise FileNotFoundError(f"Gagal menemukan checkpoint di lokal atau HF: {e}")

    # 3. Load Config & Inisialisasi Model
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Gunakan globals() atau __getattr__ secara eksplisit
    model_name = config['name']
    model_class = __getattr__(model_name)
    
    # Gabungkan argument config dengan kwargs baru
    model_args = {**config.get('args', {}), **kwargs}
    
    # Buat model langsung di device & dtype yang benar (Hemat Memori!)
    model = model_class(**model_args)
    
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    model = model.to(device=device)

    # 4. Load Weights
    state_dict = load_file(model_file, device=device)
    model.load_state_dict(state_dict, strict=False)
    
    print(f"[*] Berhasil memuat {model_name} (Ganesha 600M)!")
    return model