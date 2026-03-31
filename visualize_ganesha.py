import os
import torch
import numpy as np
import trimesh
from skimage import measure, filters
import matplotlib.pyplot as plt

def main():
    # 1. SETUP PATH & DEVICE
    output_dir = "hasil_rekonstruksi_pande"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_path = "ganesha_latent_result.pt" 
    
    if not os.path.exists(latent_path):
        print(f"[!] Error: File {latent_path} tidak ditemukan!")
        return

    print(f"[*] Memproses data laten: {latent_path}")
    
    # 2. LOAD & PREPROCESS DATA
    latent = torch.load(latent_path, map_location=device)
    
    # Konversi ke 3D Voxel Grid (Mengambil rata-rata atau norm dari channels)
    if latent.dim() == 5: # (B, C, D, H, W)
        # Kita ambil Magnitude dari semua feature channels
        voxel_grid = torch.norm(latent.squeeze(0), dim=0).cpu().numpy()
    else:
        voxel_grid = latent.cpu().numpy()

    # Normalisasi ke rentang 0.0 - 1.0
    v_min, v_max = voxel_grid.min(), voxel_grid.max()
    voxel_grid = (voxel_grid - v_min) / (v_max - v_min + 1e-8)

    # 3. PANDE-SMOOTHING ENGINE
    # Step A: Sigmoid Contrast (Mempertegas batas objek)
    # Semakin tinggi angka 15, semakin 'keras' batas antara udara dan batu
    print("[*] Menerapkan Sigmoid Contrast...")
    voxel_grid = 1 / (1 + np.exp(-15 * (voxel_grid - 0.5)))

    # Step B: Gaussian Blur (Menghilangkan efek kotak/voxelated)
    # Sigma 0.8 - 1.2 adalah sweet spot untuk resolusi 32
    print("[*] Menghaluskan permukaan (Gaussian Blur)...")
    voxel_grid = filters.gaussian(voxel_grid, sigma=1.0)

    # 4. MARCHING CUBES (EKSTRAKSI MESH)
    # Jika hasilnya masih kotak, kita naikkan level ke 0.7 atau 0.8
    # Ini ibarat kita mengupas kulit luar kubus untuk mencari isinya
    levels_to_try = [0.5, 0.6, 0.7, 0.8]
    
    for lv in levels_to_try:
        print(f"[*] Mencoba ekstraksi dengan Threshold Level: {lv}...")
        try:
            verts, faces, normals, values = measure.marching_cubes(voxel_grid, level=lv)
            
            # Center the model
            verts -= np.mean(verts, axis=0)
            
            # Create Mesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)

            # Step C: Laplacian Smoothing (Final Polish)
            trimesh.smoothing.filter_laplacian(mesh, iterations=10)

            # Export
            filename = f"Ganesha_lv_{lv}.obj"
            output_path = os.path.join(output_dir, filename)
            mesh.export(output_path)
            print(f"    > BERHASIL: {output_path}")

        except Exception as e:
            print(f"    > GAGAL pada level {lv}: {e}")

    print("\n" + "="*40)
    print("PROSES SELESAI!")
    print(f"Cek folder '{output_dir}'")
    print("Saran: Gunakan file dengan level tertinggi (0.8) jika ingin bentuk terkecil.")
    print("="*40)

if __name__ == "__main__":
    main()