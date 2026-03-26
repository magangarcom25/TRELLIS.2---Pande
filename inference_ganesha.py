import os
import sys
import torch
from PIL import Image

# 1. Daftarkan root project agar module trellis2 bisa di-import
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

# 2. Import Pipeline yang benar sesuai hasil 'ls' kamu
from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Menggunakan device: {device}")

    # 3. Path Checkpoint Fine-Tuning Ganesha Pande
    # Kita gunakan yang di ganesha_final karena log-nya paling mantap
    ckpt_path = "experiments/ganesha_final/ckpts/denoiser_step0005000.pt"
    
    if not os.path.exists(ckpt_path):
        # Fallback jika folder ganesha_final belum selesai sinkronisasi
        ckpt_path = "experiments/ganesha_reconstruction_v1/ckpts/denoiser_step0005000.pt"

    print(f"[*] Memuat Checkpoint Ganesha: {ckpt_path}")

    # 4. Inisialisasi Pipeline
    # Kita load model dasar dulu, lalu timpa dengan hasil fine-tuning kamu
    print("[*] Menginisialisasi Trellis2 Pipeline...")
    pipe = Trellis2ImageTo3DPipeline.from_pretrained("nvidia/trellis-image-to-3d").to(device)
    
    # Suntikkan bobot hasil training batu padas kamu
    print("[*] Menyuntikkan hasil fine-tuning ke model...")
    state_dict = torch.load(ckpt_path, map_location=device)
    pipe.model.load_state_dict(state_dict, strict=False)
    
    # 5. Load Gambar Test
    img_path = "data/test_images/Ganesha.jpg"
    if not os.path.exists(img_path):
        print(f"[!] Error: File {img_path} tidak ditemukan!")
        return
        
    image = Image.open(img_path).convert('RGB')
    print(f"[*] Memproses Gambar: {img_path}")

    # 6. Jalankan Rekonstruksi 3D
    print("[*] Sedang merekonstruksi... (Proses ini butuh waktu ~20 detik di GPU)")
    with torch.no_grad():
        # Kita gunakan 50 steps agar detail ukiran batunya tajam
        outputs = pipe(image, num_inference_steps=50)

    # 7. Simpan ke .PLY
    output_filename = "ganesha_3d_pande_final.ply"
    
    # Simpan sebagai Gaussian Splats (format paling detail untuk tekstur batu)
    outputs.save_ply(output_filename)
    
    print("\n" + "="*50)
    print(f" [!] BERHASIL, PANDE!")
    print(f" File 3D: {os.path.abspath(output_filename)}")
    print("="*50)

if __name__ == "__main__":
    main()