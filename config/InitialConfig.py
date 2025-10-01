import sys
import os
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib
else:
    import urllib

try:
    from imageio import imsave
except:
    try:
        from PIL import Image

        def imsave(filename, image, format=None):
            Image.fromarray(image).save(filename)
    except:
        from scipy.misc import imsave


class InitialConfig:
    HEIGHT = 96
    WIDTH = 96
    DEPTH = 3
    SIZE = HEIGHT * WIDTH * DEPTH

    CLASS_NAMES = [
        'airplane', 'bird', 'car', 'cat', 'deer',
        'dog', 'horse', 'monkey', 'ship', 'truck'
    ]

    def __init__(self, data_path: str = "./data"):
        self.dataset_url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
        self.download_path = os.path.abspath(data_path)
        self.final_path = os.path.join(self.download_path, "stl10_binary")

        self.unlabeled_path = os.path.join(self.final_path, "unlabeled_X.bin")
        self.train_images_path = os.path.join(self.final_path, "train_X.bin")
        self.train_labels_path = os.path.join(self.final_path, "train_y.bin")
        self.test_images_path = os.path.join(self.final_path, "test_X.bin")
        self.test_labels_path = os.path.join(self.final_path, "test_y.bin")
        self.class_names_path = os.path.join(
            self.final_path, "class_names.txt")

        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

    def start(self) -> str:
        filename = self.dataset_url.split('/')[-1]
        filepath = os.path.join(self.download_path, filename)

        if not os.path.exists(filepath):
            print(f"Descargando dataset desde {self.dataset_url}...")

            def _progress(count, block_size, total_size):
                percent = float(count * block_size) / float(total_size) * 100.0
                sys.stdout.write(f'\rDescargando {filename} {percent:.2f}%')
                sys.stdout.flush()

            filepath, _ = urllib.urlretrieve(
                self.dataset_url,
                filepath,
                reporthook=_progress
            )
            print(f'\nDescargado: {filename}')
        else:
            print(f"Archivo ya existe: {filepath}")

        if not os.path.exists(self.final_path):
            print(f"Extrayendo archivos...")
            tarfile.open(filepath, 'r:gz').extractall(self.download_path)
            print(f"Extraído en: {self.final_path}")
        else:
            print(f"Directorio ya existe: {self.final_path}")

        return self.final_path

    def read_labels(self, path_to_labels):
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            return labels

    def read_all_images(self, path_to_data):
        with open(path_to_data, 'rb') as f:
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 3, 2, 1))
            return images

    def read_single_image(self, image_file):
        image = np.fromfile(image_file, dtype=np.uint8, count=self.SIZE)
        image = np.reshape(image, (3, 96, 96))
        image = np.transpose(image, (2, 1, 0))
        return image

    def get_class_names(self):
        if os.path.exists(self.class_names_path):
            with open(self.class_names_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            return class_names
        return self.CLASS_NAMES

    def load_unlabeled(self) -> np.ndarray:
        print("Loading unlabeled split (100,000 images)...")
        images = self.read_all_images(self.unlabeled_path)
        print(f"Loaded unlabeled: {images.shape}")
        return images

    def load_train(self, with_labels: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        print("Loading train split (5,000 images)...")
        images = self.read_all_images(self.train_images_path)

        labels = None
        if with_labels:
            labels = self.read_labels(self.train_labels_path)
            print(f"Loaded train: {images.shape}, labels: {labels.shape}")
        else:
            print(f"Loaded train: {images.shape} (without labels)")

        return images, labels

    def load_test(self, with_labels: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        print("Loading test split (8,000 images)...")
        images = self.read_all_images(self.test_images_path)

        labels = None
        if with_labels:
            labels = self.read_labels(self.test_labels_path)
            print(f"Loaded test: {images.shape}, labels: {labels.shape}")
        else:
            print(f"Loaded test: {images.shape} (without labels)")

        return images, labels

    def load_all_for_training(self) -> np.ndarray:
        print("Loading all images for unsupervised training...")
        unlabeled = self.load_unlabeled()
        train_images, _ = self.load_train(with_labels=False)
        test_images, _ = self.load_test(with_labels=False)

        all_images = np.concatenate(
            [unlabeled, train_images, test_images], axis=0)
        print(f"Total images for training: {all_images.shape}")
        return all_images

    def load_for_evaluation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_images, train_labels = self.load_train(with_labels=True)
        test_images, test_labels = self.load_test(with_labels=True)
        return train_images, train_labels, test_images, test_labels

    def get_class_name(self, label: int) -> str:
        if 1 <= label <= 10:
            return self.CLASS_NAMES[label - 1]
        return "unknown"

    def get_label_distribution(self, labels: np.ndarray) -> dict:
        unique, counts = np.unique(labels, return_counts=True)
        distribution = {}

        for label, count in zip(unique, counts):
            class_name = self.get_class_name(label)
            distribution[class_name] = count

        return distribution

    def plot_image(self, image, label=None, class_names=None):
        plt.imshow(image)
        if label is not None and class_names is not None:
            plt.title(f"Clase: {class_names[label-1]}")
        elif label is not None:
            plt.title(f"Label: {label}")
        plt.axis('off')
        plt.show()

    def plot_samples(self, images, labels, class_names=None, n_samples=16):
        if class_names is None:
            class_names = self.CLASS_NAMES

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('Muestras del dataset STL-10', fontsize=16)

        for i, ax in enumerate(axes.flat):
            if i < n_samples and i < len(images):
                ax.imshow(images[i])
                ax.set_title(f"{class_names[labels[i]-1]}", fontsize=10)
                ax.axis('off')

        plt.tight_layout()
        plt.show()

    def save_images_to_disk(self, images, labels, output_dir='./img'):
        print("Guardando imágenes en disco...")

        for i, (image, label) in enumerate(zip(images, labels)):
            directory = os.path.join(output_dir, str(label))
            os.makedirs(directory, exist_ok=True)

            filename = os.path.join(directory, f"{i}.png")
            imsave(filename, image, format="png")

            if i % 100 == 0:
                print(f"Guardadas {i}/{len(images)} imágenes...")

        print(f"¡Completado! Imágenes guardadas en {output_dir}")
