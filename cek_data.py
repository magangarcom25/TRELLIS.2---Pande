import trimesh
import os
import numpy as np

def audit_dataset():
    # Sesuaikan path dengan folder punyamu
    dataset_path = "datasets/Ganesha_Dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Folder {dataset_path} tidak ditemukan!")
        return

    # Ambil semua folder yang diawali kata Ganesha
    folders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f)) and f.startswith('Ganesha')])

    print(f"🔍 Memulai Audit Mendalam pada {len(folders)} Model Arca Ganesha...\n")
    print(f"{'Folder':<20} | {'Faces':<10} | {'Vertices':<10} | {'Scale (Max)':<10} | {'Status'}")
    print("-" * 75)

    error_count = 0
    warning_count = 0

    for folder in folders:
        file_path = os.path.join(dataset_path, folder, "model.glb")
        status = "✅ OK"
        
        if not os.path.exists(file_path):
            print(f"{folder:<20} | {'-':<10} | {'-':<10} | {'-':<10} | ❌ FILE HILANG")
            error_count += 1
            continue

        try:
            # Muat mesh (force_mesh agar tidak dalam bentuk scene)
            mesh_data = trimesh.load(file_path, force='mesh')
            
            faces = len(mesh_data.faces)
            verts = len(mesh_data.vertices)
            
            # Cek Dimensi (Bounding Box)
            # Penting: TRELLIS biasanya butuh model di dalam range -0.5 sampai 0.5 atau -1 sampai 1
            extents = mesh_data.extents
            max_dim = max(extents)
            
            # 1. Cek apakah model kosong
            if faces == 0:
                status = "❌ KOSONG"
                error_count += 1
            # 2. Cek apakah terlalu low-poly (detail hilang)
            elif faces < 500:
                status = "⚠️ LOW-DETAIL"
                warning_count += 1
            # 3. Cek apakah skala terlalu besar/kecil (menyebabkan 'debu' saat training)
            elif max_dim > 10.0 or max_dim < 0.1:
                status = "⚠️ SCALE ISSUE"
                warning_count += 1
            
            print(f"{folder:<20} | {faces:<10} | {verts:<10} | {max_dim:<10.2f} | {status}")

        except Exception as e:
            print(f"{folder:<20} | {'-':<10} | {'-':<10} | {'-':<10} | ❌ CORRUPT: {str(e)[:15]}")
            error_count += 1

    print("-" * 75)
    print(f"📊 HASIL AUDIT:")
    print(f"   - Total Folder: {len(folders)}")
    print(f"   - Error (Harus Diperbaiki): {error_count}")
    print(f"   - Warning (Perlu Dicek): {warning_count}")
    
    if error_count == 0:
        print("\n🚀 KESIMPULAN: Datasetmu SEHAT. Masalah kemungkinan besar di konfigurasi Fine-Tuning atau Inference.")
    else:
        print("\n🆘 KESIMPULAN: Ada masalah pada dataset. AI tidak bisa belajar dari file yang error/kosong.")

if __name__ == "__main__":
    audit_dataset()