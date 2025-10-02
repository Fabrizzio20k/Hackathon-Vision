#!/usr/bin/env python3

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from config.InitialConfig import InitialConfig
from utils.RegionalMultiDescriptor import RegionalMultiDescriptor

# Cargar datos
config = InitialConfig(data_path="./dataset")
config.start()

print("Loading data...")
train_images, train_labels = config.load_train(with_labels=True)
test_images, test_labels = config.load_test(with_labels=True)

# Usar subset para prueba rápida
train_subset = 2000
test_subset = 2000

# Extraer features
print("Extracting features...")
descriptor = RegionalMultiDescriptor(grid_size=2)
train_features = np.array([descriptor.extract(img) for img in train_images[:train_subset]])
test_features = np.array([descriptor.extract(img) for img in test_images[:test_subset]])

# Probar diferentes valores de C
C_values = [0.1, 1.0, 10.0, 100.0, 1000.0]

for C in C_values:
    print(f"\nTesting C={C}...")
    clf = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
    clf.fit(train_features, train_labels[:train_subset])
    pred = clf.predict(test_features)
    acc = accuracy_score(test_labels[:test_subset], pred)
    print(f"Accuracy: {acc:.4f}")