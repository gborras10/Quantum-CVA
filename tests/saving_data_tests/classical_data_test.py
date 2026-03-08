import numpy as np
import pathlib

# Ruta al archivo guardado
path = pathlib.Path(__file__).resolve().parents[2] / "data" / "multi_asset" / "benchmark" / "three_asset_instance.npz"

# Cargar el archivo
data = np.load(path, allow_pickle=True)

# Mostrar todas las claves y sus valores
print("Datos guardados en el archivo:")
for key in data.files:
    print(f"\nClave: {key}")
    print(data[key])
