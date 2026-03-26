import importlib
from typing import Any

# Daftar pemetaan class ke file modulnya
__attributes = {
    # --- Custom Feature Extractors (TAMBAHKAN DI SINI) ---
    'DinoV2FeatureExtractor': 'trainers.flow_matching.mixins.image_conditioned',
    'DinoV3FeatureExtractor': 'trainers.flow_matching.mixins.image_conditioned',
    
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
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_path = __attributes[name]
            
            # Jika modul berada di luar folder models (seperti mixins), kita handle path-nya
            if 'trainers.' in module_path:
                # Import dari trellis2.trainers...
                full_module_path = f"trellis2.{module_path}"
                module = importlib.import_module(full_module_path)
            else:
                # Import relatif standar untuk internal models
                module = importlib.import_module(f".{module_path}", __name__)
                
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.
    """
    import os
    import json
    from safetensors.torch import load_file
    
    # Cek apakah path adalah file lokal atau HF
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Memanggil __getattr__ untuk inisialisasi class
    model_class = __getattr__(config['name'])
    model = model_class(**config['args'], **kwargs)
    model.load_state_dict(load_file(model_file), strict=False)

    return model


# Untuk Pylance / IDE Autocomplete
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder