import os
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from skimage import measure, filters

def main():
    # --- 1. SETUP & LOAD ---
    latent_path = "ganesha_latent_result.pt"
    output_dir = "hasil_final_pande"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(latent_path):
        print(f"[!] Error: File {latent_path} tidak ditemukan!")
        return

    print(f"[*] Menganalisis & Visualisasi: {latent_path}")
    latent = torch.load(latent_path, map_location=device)
    
    # Konversi ke numpy untuk analisis statistik
    if isinstance(latent, torch.Tensor):
        data_flat = latent.detach().cpu().numpy().flatten()
        # Ambil voxel grid (B, C, D, H, W) -> (D, H, W)
        if latent.dim() == 5:
            # Menggunakan norm untuk menggabungkan feature channels menjadi bentuk fisik 3D
            voxel_grid = torch.norm(latent.squeeze(0), dim=0).cpu().numpy()
        else:
            voxel_grid = latent.cpu().numpy()
    else:
        data_flat = np.array(latent).flatten()
        voxel_grid = np.array(latent)

    # --- 2. TAHAP DIAGNOSA (STATISTIK) ---
    v_min, v_max = data_flat.min(), data_flat.max()
    v_std = data_flat.std()
    
    print("-" * 40)
    print(f"DIAGNOSA DATA:")
    print(f"  > Min: {v_min:.4f} | Max: {v_max:.4f}")
    print(f"  > Standar Deviasi: {v_std:.4f}")
    
    if v_std < 0.05:
        print("[ALERT] Data flat! Model mungkin mengabaikan foto input.")
    else:
        print("[OK] Sinyal objek terdeteksi. Memulai ekstraksi bentuk...")
    print("-" * 40)

    # --- 3. TAHAP HISTOGRAM (DISTRIBUSI) ---
    plt.figure(figsize=(8, 4))
    plt.hist(data_flat, bins=100, color='#50C878', alpha=0.7, edgecolor='black')
    plt.title("Distribusi Nilai Laten Ganesha (Step 5000)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plot_path = os.path.join(output_dir, "distribusi_ganesha.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[*] Grafik distribusi diperbarui: {plot_path}")

    # --- 4. TAHAP SMOOTHING & MESHING (OPTIMIZED) ---
    # Normalisasi voxel grid ke 0-1
    voxel_grid = (voxel_grid - voxel_grid.min()) / (voxel_grid.max() - voxel_grid.min() + 1e-8)
    
    # [TUNING] Gain 12 & Sigma 1.2 untuk hasil arca yang lebih solid (tidak pecah)
    print("[*] Menerapkan Pande-Tuning (Gain: 12, Sigma: 1.2)...")
    voxel_grid = 1 / (1 + np.exp(-12 * (voxel_grid - 0.5)))
    voxel_grid = filters.gaussian(voxel_grid, sigma=1.2)

    # Mengekstraksi Mesh dengan 3 variasi level ketebalan
    for lv in [0.4, 0.6, 0.8]:
        print(f"[*] Mengekstraksi Mesh (Level {lv})...")
        try:
            # Marching Cubes
            verts, faces, _, _ = measure.marching_cubes(voxel_grid, level=lv)
            
            # Centering agar objek muncul tepat di tengah grid Blender
            verts -= np.mean(verts, axis=0)
            
            # Membuat objek Trimesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            
            # Laplacian Smoothing untuk menghaluskan voxel menjadi permukaan halus
            trimesh.smoothing.filter_laplacian(mesh, iterations=15)
            
            out_file = os.path.join(output_dir, f"Ganesha_Final_LV{lv}.obj")
            mesh.export(out_file)
            print(f"  > File tersimpan: {out_file}")
            
        except Exception as e:
            print(f"  > Gagal di Level {lv}: {e}")

    print("\n" + "="*40)
    print("[SUCCESS] Silakan cek folder: hasil_final_pande")
    print("Saran: Buka Ganesha_Final_LV0.4.obj di Blender!")
    print("="*40)

if __name__ == "__main__":
    main()