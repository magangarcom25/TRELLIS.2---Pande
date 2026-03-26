import trellis2.models as models

print("--- DAFTAR MODEL YANG TERSEDIA DI TRELLIS2 ---")
try:
    # Mengecek isi dari registry model
    if hasattr(models, 'api'):
        print(models.api.models.keys())
    elif hasattr(models, 'renderers'):
        print("Mengecek attribute langsung di models:")
        print([attr for attr in dir(models) if not attr.startswith('_')])
    else:
        print("Mencoba list dir models:")
        print(dir(models))
except Exception as e:
    print(f"Gagal mengecek: {e}")