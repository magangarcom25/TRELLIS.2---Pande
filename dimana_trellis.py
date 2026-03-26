import trellis2
import os
print(f"Lokasi Library: {os.path.dirname(trellis2.__file__)}")
print("\nIsi folder models:")
models_path = os.path.join(os.path.dirname(trellis2.__file__), 'models')
if os.path.exists(models_path):
    print(os.listdir(models_path))
else:
    print("Folder models tidak ditemukan di lokasi tersebut!")