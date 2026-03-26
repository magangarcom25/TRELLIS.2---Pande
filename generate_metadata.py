import os
import csv

def generate_metadata():
    # Sesuaikan dengan path dataset Ganesha kamu
    base_dir = "/home/pande/TRELLIS.2---Pande/datasets/Ganesha_Dataset"
    output_file = os.path.join(base_dir, "metadata.csv")
    
    # Ambil semua folder Ganesha yang sudah berisi file render
    folders = [f for f in os.listdir(base_dir) if f.startswith('Ganesha') and os.path.isdir(os.path.join(base_dir, f))]
    folders.sort() # Mengurutkan nama folder

    # Menulis ke file metadata.csv
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header standard untuk dataset TRELLIS
        writer.writerow(['id', 'prompt'])
        
        for folder in folders:
            # Gunakan nama folder sebagai ID dan deskripsi umum sebagai prompt
            writer.writerow([folder, "A statue of Ganesha"])
            
    print(f"✅ Metadata berhasil dibuat! {len(folders)} folder terdaftar.")

if __name__ == "__main__":
    generate_metadata()
