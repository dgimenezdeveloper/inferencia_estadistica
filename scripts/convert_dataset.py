import pandas as pd

# Carga el archivo txt (separado por espacios o tabulaciones)
df = pd.read_csv('../datos/seeds_dataset.txt', sep='\s+', header=None)

# Asigna nombres de columnas (según la documentación UCI)
df.columns = [
    'area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'groove', 'class'
]

# Guarda como CSV
df.to_csv('seeds_dataset.csv', index=False)