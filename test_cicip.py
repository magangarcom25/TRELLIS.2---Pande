import numpy as np
import os

# Alamat folder harta karun
folder_path = 'datasets/sslat/shape_latents/shape_enc_next_dc_f16c32_fp16_1024'
file_target = 'Ganesha10.npz' # Kita coba yang muncul di ls tadi
full_path = os.path.join(folder_path, file_target)

try:
    # Load file .npz
    data = np.load(full_path, allow_pickle=True)
    
    print("-" * 40)
    print(f"✅ BERHASIL LOAD: {file_target}")
    print("-" * 40)
    
    # Intip isi di dalam file .npz
    for key in data.files:
        val = data[key]
        print(f"🔑 Key: {key}")
        print(f"   📊 Shape: {val.shape}")
        print(f"   🧬 Type: {val.dtype}")
    
    print("-" * 40)
    print("Kesimpulan: Data Ganesha aman dalam format NumPy!")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
