import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Optional
import pickle
from pathlib import Path
from tqdm import tqdm


class SIFTBoVW:
    def __init__(
        self,
        n_clusters: int = 500,
        sift_features: int = 0,
        nOctaveLayers: int = 3,
        contrastThreshold: float = 0.04,
        edgeThreshold: int = 10
    ):
        self.n_clusters = n_clusters
        self.sift = cv2.SIFT_create(
            nfeatures=sift_features,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold
        )
        self.kmeans = None
        self.is_trained = False

    def extract_sift_descriptors(self, image: np.ndarray) -> Optional[np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        _, descriptors = self.sift.detectAndCompute(gray, None)

        return descriptors

    def extract_sift_from_images(
        self,
        images: np.ndarray,
        max_descriptors_per_image: Optional[int] = None
    ) -> np.ndarray:
        all_descriptors = []

        for img in tqdm(images, desc="Extracting SIFT"):
            desc = self.extract_sift_descriptors(img)

            if desc is not None and len(desc) > 0:
                if max_descriptors_per_image:
                    desc = desc[:max_descriptors_per_image]
                all_descriptors.append(desc)

        if len(all_descriptors) == 0:
            raise ValueError("No SIFT descriptors found in images")

        return np.vstack(all_descriptors)

    def train_codebook(
        self,
        images: np.ndarray,
        sample_size: Optional[int] = None,
        max_descriptors_per_image: Optional[int] = 100
    ):
        print(f"Training BoVW with {self.n_clusters} clusters...")

        if sample_size and sample_size < len(images):
            indices = np.random.choice(len(images), sample_size, replace=False)
            images = images[indices]

        all_descriptors = self.extract_sift_from_images(
            images,
            max_descriptors_per_image
        )

        print(f"Total descriptors collected: {len(all_descriptors)}")

        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=1000,
            max_iter=100,
            random_state=42,
            verbose=1
        )

        self.kmeans.fit(all_descriptors)
        self.is_trained = True

        print("Codebook training complete!")

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError(
                "Codebook not trained. Call train_codebook first.")

        descriptors = self.extract_sift_descriptors(image)

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_clusters, dtype=np.float32)

        labels = self.kmeans.predict(descriptors)

        histogram = np.bincount(labels, minlength=self.n_clusters)
        histogram = histogram.astype(np.float32)

        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()

        return histogram

    def encode_images(self, images: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError(
                "Codebook not trained. Call train_codebook first.")

        vectors = []

        for img in tqdm(images, desc="Encoding images"):
            vec = self.encode_image(img)
            vectors.append(vec)

        return np.array(vectors, dtype=np.float32)

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        state = {
            'n_clusters': self.n_clusters,
            'kmeans': self.kmeans,
            'is_trained': self.is_trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        extractor = cls(n_clusters=state['n_clusters'])
        extractor.kmeans = state['kmeans']
        extractor.is_trained = state['is_trained']

        return extractor

    def get_feature_dimension(self) -> int:
        return self.n_clusters
