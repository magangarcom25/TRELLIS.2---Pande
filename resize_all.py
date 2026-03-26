import os
from PIL import Image

root_dir = "./datasets/Ganesha_Dataset"
target_size = (256, 256)

print(f"Mulai memproses folder: {root_dir}")

for root, dirs, files in os.walk(root_dir):
    for filename in files:
        if filename.lower().endswith(".png"):
            file_path = os.path.join(root, filename)
            try:
                with Image.open(file_path) as img:
                    if img.size != target_size:
                        # Menggunakan LANCZOS untuk menjaga kualitas pahatan Ganesha
                        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                        img_resized.save(file_path)
                        print(f"✅ Resized: {file_path}")
            except Exception as e:
                print(f"❌ Error pada {file_path}: {e}")

print("--- Selesai! Semua gambar di semua sub-folder kini 256x256. ---")
