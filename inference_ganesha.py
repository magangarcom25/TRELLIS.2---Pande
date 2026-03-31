import os
import sys
import torch
import json
import numpy as np
from PIL import Image
from collections import OrderedDict
import spconv.pytorch as spconv

# ==============================================================================
# 1. ENVIRONMENT & MONKEY PATCH (Agar Spconv lancar di LAB-AI-04)
# ==============================================================================
os.environ['SPARSE_CONV_BACKEND'] = 'spconv'
os.environ['SPARSE_BACKEND'] = 'spconv'

# Patch untuk kompatibilitas atribut spconv
spconv.SparseConvTensor.feats = property(lambda self: self.features)
spconv.SparseConvTensor.coords = property(lambda self: self.indices)
spconv.SparseConvTensor.shape = property(lambda self: self.spatial_shape)

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
from feature_extractors import DinoV2FeatureExtractor

# ==============================================================================
# 2. PROSES SPARSITY (Kunci Penghilang Box)
# ==============================================================================
def tensor_to_sparse(x, device, threshold=0.05):
    """
    Mengonversi tensor padat ke sparse. 
    Threshold 0.05 akan membuang 'noise' di pinggiran kotak.
    """
    B, C, D, H, W = x.shape
    # Masking: Hanya ambil voxel yang memiliki intensitas sinyal tertentu
    mask = torch.abs(x).mean(dim=1) > threshold 
    indices = torch.where(mask)
    
    if indices[0].numel() == 0: # Fallback jika terlalu bersih
        indices = torch.where(torch.ones_like(mask, dtype=torch.bool))

    b, d, h, w = indices
    coords = torch.stack([b, d, h, w], dim=-1).int().to(device)
    feats = x[b, :, d, h, w]
    
    return spconv.SparseConvTensor(feats, coords, [D, H, W], B)

# ==============================================================================
# 3. MAIN INFERENCE
# ==============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Menjalankan pada: {device}")

    # --- Load Config ---
    config_path = "configs/ganesha_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # --- Inisialisasi Arsitektur ---
    model = SparseStructureFlowModel(
        resolution=config.get('resolution', 32),
        in_channels=config.get('in_channels', 8),
        model_channels=config.get('model_channels', 128),
        out_channels=config.get('out_channels', 32),
        num_res_blocks=config.get('num_res_blocks', 12)
    ).to(device)

    # --- Load Weights (Step 5000) ---
    ckpt_path = "experiments/ganesha_reconstruction_v1/ckpts/denoiser_step0005000.pt"
    if not os.path.exists(ckpt_path):
        print(f"[!] Checkpoint {ckpt_path} tidak ditemukan!")
        return

    state_dict = torch.load(ckpt_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # --- DINOv2 (ViT) Feature Extraction ---
    print("[*] Memuat DINOv2 untuk ekstraksi fitur Arca...")
    feature_extractor = DinoV2FeatureExtractor("dinov2_vitb14").to(device).eval()

    # --- Load Image Ganesha ---
    img_path = "data/test_images/Ganesha.jpg"
    image = Image.open(img_path).convert('RGB').resize((518, 518))
    img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    # --- Sampling Loop (Flow Matching) ---
    print("[*] Memahat Arca Ganesha (Anti-Box Mode)...")
    with torch.no_grad():
        # Fitur panduan dari foto
        cond = feature_extractor(img_tensor)
        
        res = config.get('resolution', 32)
        out_channels = config.get('out_channels', 32)
        
        # Inisialisasi Latent (Start from Noise)
        x_dense = torch.randn(1, out_channels, res, res, res).to(device)
        
        steps = 50
        dt = 1.0 / steps
        
        for i in range(steps):
            t_val = i / steps
            t = torch.full((1,), t_val, device=device)
            
            # Konversi ke Sparse dengan Thresholding
            x_sparse = tensor_to_sparse(x_dense, device, threshold=0.1)
            
            # Prediksi Velocity (Aliran bentuk)
            v_sparse = model(x_sparse, t, cond)
            v_dense = v_sparse.to_dense() if hasattr(v_sparse, 'to_dense') else v_sparse.dense()
            
            # --- Classifier-Free Guidance (Kunci Detail) ---
            # Skala 4.0 sangat kuat memaksa AI mengikuti siluet foto
            cfg_scale = 4.0 
            x_dense = x_dense + (v_dense * dt * cfg_scale)
            
            if i % 10 == 0:
                print(f"    > Step {i}/50: Mengupas lapisan kotak...")

    # --- Final Polish ---
    x_dense = torch.clamp(x_dense, -1.0, 1.0)
    # Normalisasi akhir
    x_dense = (x_dense - x_dense.min()) / (x_dense.max() - x_dense.min() + 1e-8)

    # Simpan hasil untuk di-visualisasi oleh pande_master_viz.py
    output_latent = "ganesha_latent_result.pt"
    torch.save(x_dense.cpu(), output_latent)
    print(f"\n[SUCCESS] File {output_latent} siap. Jalankan visualisasi sekarang!")

if __name__ == "__main__":
    main()