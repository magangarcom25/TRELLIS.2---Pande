import os
import json

# Alamat folder dataset kamu
dataset_path = "datasets/Ganesha_Dataset"
output_file = "datasets/ganesha_manifest.jsonl"

# Ambil semua folder Ganesha yang ada
folders = sorted([f for f in os.listdir(dataset_path) if f.startswith('Ganesha')])

print(f"🚀 Sedang membuat manifest untuk 41 folder Ganesha...")

count = 0
with open(output_file, 'w') as f:
    for folder in folders:
        # Kita lewati Ganesha1 jika memang korup, jika sudah oke hapus baris if ini
        if folder == "Ganesha1":
            print(f"⚠️ Melewati {folder} karena file terdeteksi korup.")
            continue
            
        # Format manifest agar TRELLIS tahu di mana folder gambarnya berada
        data = {
            "id": folder,
            "prompt": "A statue of Ganesha",
            "path": f"Ganesha_Dataset/{folder}" 
        }
        f.write(json.dumps(data) + '\n')
        count += 1

print(f"\n✅ Selesai! Manifest dibuat dengan {count} data.")
print(f"📍 Lokasi file: {output_file}")