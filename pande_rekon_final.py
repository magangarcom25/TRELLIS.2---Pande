import os
import torch
import numpy as np
import trimesh
from skimage import measure, filters
import matplotlib.pyplot as plt

def main():
    # 1. SETUP PATH & DEVICE
    # Kita buat folder khusus agar hasil lama tidak tertukar
    output_dir = "hasil_rekonstruksi_pande_V4"
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
    
    # Konversi ke 3D Voxel Grid
    if isinstance(latent, torch.Tensor):
        if latent.dim() == 5: # (B, C, D, H, W)
            # Kita ambil Magnitude (Norm) untuk menggabungkan 32 channel menjadi 1 bentuk fisik
            voxel_grid = torch.norm(latent.squeeze(0), dim=0).cpu().numpy()
        else:
            voxel_grid = latent.cpu().numpy()
    else:
        voxel_grid = np.array(latent)

    # Normalisasi ke rentang 0.0 - 1.0 (Penting agar Thresholding akurat)
    v_min, v_max = voxel_grid.min(), voxel_grid.max()
    voxel_grid = (voxel_grid - v_min) / (v_max - v_min + 1e-8)

    # 3. PANDE-SMOOTHING ENGINE (Penghilang Box & Lubang)
    # Step A: Sigmoid Contrast (Tuning: 10.0 - Angka lebih rendah agar tidak terlalu tajam)
    print("[*] Menerapkan Sigmoid Contrast (Lembut)...")
    voxel_grid = 1 / (1 + np.exp(-10 * (voxel_grid - 0.5)))

    # Step B: Gaussian Blur (Tuning: 1.2 - Untuk 'menjahit' lubang-lubang di mesh)
    print("[*] Menghaluskan permukaan (Gaussian Blur)...")
    voxel_grid = filters.gaussian(voxel_grid, sigma=1.2)

    # 4. MARCHING CUBES (EKSTRAKSI MESH)
    # Kita coba level dari 0.4 sampai 0.8
    levels_to_try = [0.4, 0.5, 0.6, 0.7, 0.8]
    
    print("-" * 40)
    for lv in levels_to_try:
        print(f"[*] Mencoba ekstraksi dengan Threshold Level: {lv}...")
        try:
            # Algoritma Marching Cubes untuk membuat permukaan 3D
            verts, faces, normals, values = measure.marching_cubes(voxel_grid, level=lv)
            
            # Memindahkan posisi ke tengah (Centering)
            verts -= np.mean(verts, axis=0)
            
            # Membuat Mesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)

            # Step C: Laplacian Smoothing (Final Polish agar tidak terlihat voxelated/kotak)
            # 15 Iterasi untuk hasil yang sangat halus seperti batu arca aslinya
            trimesh.smoothing.filter_laplacian(mesh, iterations=15)

            # Export ke format OBJ (Bisa dibuka di Blender)
            filename = f"Ganesha_Final_Pande_LV{lv}.obj"
            output_path = os.path.join(output_dir, filename)
            mesh.export(output_path)
            print(f"    > BERHASIL: {output_path}")

        except Exception as e:
            print(f"    > GAGAL pada level {lv}: {e}")

    print("\n" + "="*40)
    print("PROSES SELESAI!")
    print(f"Silakan cek folder: '{output_dir}'")
    print("-" * 40)
    print("INFO UNTUK PANDE:")
    print("1. Ganesha_Final_Pande_LV0.4.obj -> Paling tebal (Saran: buka ini dulu!)")
    print("2. Ganesha_Final_Pande_LV0.8.obj -> Paling tipis (Pahatan paling dalam)")
    print("="*40)

if __name__ == "__main__":
    main()