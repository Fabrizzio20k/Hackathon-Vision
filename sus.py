import numpy as np

data = np.load('models/features.npz')

print("Archivos en el .npz:", data.files)
print("\n=== TRAIN VECTORS ===")
print(f"Shape: {data['train_vectors'].shape}")
print(f"Min: {data['train_vectors'].min():.4f}")
print(f"Max: {data['train_vectors'].max():.4f}")
print(f"Mean: {data['train_vectors'].mean():.4f}")
print(f"Std: {data['train_vectors'].std():.4f}")

print("\n=== TEST VECTORS ===")
print(f"Shape: {data['test_vectors'].shape}")
print(f"Min: {data['test_vectors'].min():.4f}")
print(f"Max: {data['test_vectors'].max():.4f}")
print(f"Mean: {data['test_vectors'].mean():.4f}")
print(f"Std: {data['test_vectors'].std():.4f}")

print("\n=== PRIMER VECTOR (5 elementos) ===")
print(data['train_vectors'][0][:5])

print("\n=== SPARSITY CHECK ===")
zeros = (data['train_vectors'] == 0).sum()
total = data['train_vectors'].size
print(f"Zeros: {zeros}/{total} ({100*zeros/total:.2f}%)")

print("\n=== DISTRIBUCIÓN POR FEATURE ===")
print("Mean por feature (primeras 10):")
print(data['train_vectors'].mean(axis=0)[:10])
print("\nStd por feature (primeras 10):")
print(data['train_vectors'].std(axis=0)[:10])