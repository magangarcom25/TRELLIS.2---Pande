import torch
import os

def check_ganesha_health():
    base_path = "experiments/ganesha_reconstruction_v1/ckpts/"
    files = {
        "denoiser": "denoiser_step0005000.pt",
        "image_cond": "image_cond_model_step0005000.pt",
        "misc": "misc_step0005000.pt"
    }

    print("="*50)
    print("🚀 GANESHA MODEL - FINAL INSPECTION (TRELLIS 2)")
    print("="*50)

    last_loss = None

    for role, filename in files.items():
        full_path = os.path.join(base_path, filename)
        print(f"\n[*] Memeriksa {role.upper()}: {filename}")
        
        if not os.path.exists(full_path):
            print(f"  [!] ERROR: File tidak ditemukan di {full_path}")
            continue

        try:
            # Load ke CPU (Aman dari OOM)
            data = torch.load(full_path, map_location='cpu')

            # 1. Cek Kesehatan Bobot (untuk Denoiser & Image Cond)
            if role in ["denoiser", "image_cond"]:
                keys = list(data.keys())
                print(f"  > Total Layer: {len(keys)}")
                
                # Cek NaN (Sanity Check)
                has_nan = False
                for k in keys[:10]: # Cek 10 layer pertama untuk kecepatan
                    if torch.isnan(data[k]).any():
                        has_nan = True
                        break
                
                if has_nan:
                    print("  [!] STATUS: ❌ TERDETEKSI NaN (Bobot rusak/meledak)")
                else:
                    print("  [✓] STATUS: ✅ Bobot Sehat (Tidak ada NaN)")

            # 2. Cek Loss & Progress (untuk Misc)
            if role == "misc":
                # Cari key loss di berbagai kemungkinan nama
                loss_keys = ['loss', 'last_loss', 'train_loss']
                for k in loss_keys:
                    if k in data:
                        last_loss = data[k]
                        break
                
                step = data.get('step', data.get('iteration', 'Unknown'))
                
                print(f"  > Step Terakhir: {step}")
                if last_loss is not None:
                    print(f"  > Loss Terakhir: {last_loss:.6f}")
                else:
                    # Jika tidak ada di root, coba cari di dalam 'optimizer'
                    print("  [?] Info: Loss tidak ditemukan di root file misc.")

        except Exception as e:
            print(f"  [!] Gagal membedah file: {e}")

    # --- ANALISIS AKHIR ---
    print("\n" + "="*50)
    print("📊 KESIMPULAN ANALISIS")
    print("="*50)
    
    if last_loss is not None:
        if last_loss < 0.05:
            print("💎 KUALITAS: SANGAT BAGUS. Detail arca seharusnya sangat tajam.")
        elif last_loss < 0.15:
            print("👍 KUALITAS: CUKUP. Bentuk Ganesha sudah solid, siap pakai.")
        else:
            print("⚠️ KUALITAS: MASIH KASAR. Disarankan tambah step fine-tuning.")
    else:
        print("💡 TIPS: Cek folder 'samples/' untuk verifikasi visual manual.")
    
    print("="*50)

if __name__ == "__main__":
    check_ganesha_health()
