import trimesh
import os

# Menuju folder dataset
dataset_path = "datasets/Ganesha_Dataset"

# PERHATIKAN: 'G' harus kapital sesuai nama foldermu di foto
folders = sorted([f for f in os.listdir(dataset_path) if f.startswith('Ganesha')])

print(f"🔍 Mulai mengecek {len(folders)} folder dataset...")

for folder in folders:
    # Memastikan mencari file model.glb di dalam folder GaneshaX
    file_path = os.path.join(dataset_path, folder, "model.glb")
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File model.glb tidak ditemukan di folder {folder}")
        continue

    try:
        # Mencoba memuat file GLB
        mesh = trimesh.load(file_path)
        # Menghitung jumlah faces/permukaan sebagai bukti file tidak kosong
        faces_count = len(mesh.faces) if hasattr(mesh, 'faces') else "N/A"
        print(f"✅ {folder}: Berhasil dimuat ({faces_count} faces)")
    except Exception as e:
        print(f"❌ ERROR pada {folder}: {e}")

print("\nSelesai! Jika ke-42 folder centang hijau, kita lanjut ke tahap Fine-Tuning.")
