import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import pickle
from pathlib import Path
from tqdm import tqdm


class DenseRootSIFT:
    def __init__(self, scales=5, stride=8, patch_size=16):
        self.scales = scales
        self.stride = stride
        self.patch_size = patch_size
        self.sift = cv2.SIFT_create()

    def extract_dense_sift(self, img_gray):
        h, w = img_gray.shape
        keypoints = []
        for y in range(0, h - self.patch_size, self.stride):
            for x in range(0, w - self.patch_size, self.stride):
                keypoints.append(cv2.KeyPoint(x, y, self.patch_size))

        _, descriptors = self.sift.compute(img_gray, keypoints)
        return descriptors

    def rootsift_normalize(self, descriptors):
        if descriptors is None:
            return None
        descriptors = descriptors / \
            (descriptors.sum(axis=1, keepdims=True) + 1e-7)
        descriptors = np.sqrt(descriptors)
        descriptors = descriptors / \
            (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-7)
        return descriptors

    def extract_multiscale(self, img_gray):
        all_descriptors = []
        for scale_idx in range(self.scales):
            scale_factor = 2 ** (scale_idx * 0.5)
            new_h = int(img_gray.shape[0] * scale_factor)
            new_w = int(img_gray.shape[1] * scale_factor)

            if new_h < self.patch_size or new_w < self.patch_size:
                continue

            img_scaled = cv2.resize(img_gray, (new_w, new_h))
            descriptors = self.extract_dense_sift(img_scaled)

            if descriptors is not None and len(descriptors) > 0:
                rootsift_desc = self.rootsift_normalize(descriptors)
                all_descriptors.append(rootsift_desc)

        if len(all_descriptors) > 0:
            return np.vstack(all_descriptors)
        return None


class FisherVectorExtractor:
    def __init__(self, n_gaussians=256, pca_dim=64, scales=5, stride=8):
        self.n_gaussians = n_gaussians
        self.pca_dim = pca_dim
        self.sift_extractor = DenseRootSIFT(scales=scales, stride=stride)
        self.pca = None
        self.gmm = None
        self.is_trained = False

    def train(self, images, sample_size=10000, max_descriptors=2000000):
        print(f"Training Fisher Vector model...")

        indices = np.random.choice(len(images), min(
            sample_size, len(images)), replace=False)
        all_descriptors = []

        for idx in tqdm(indices, desc="Extracting SIFT"):
            img = self._prepare_image(images[idx])
            descriptors = self.sift_extractor.extract_multiscale(img)
            if descriptors is not None:
                all_descriptors.append(descriptors)

        pooled = np.vstack(all_descriptors)

        if len(pooled) > max_descriptors:
            indices = np.random.choice(
                len(pooled), max_descriptors, replace=False)
            pooled = pooled[indices]

        print(f"Training PCA: 128D -> {self.pca_dim}D...")
        self.pca = PCA(n_components=self.pca_dim, whiten=True, random_state=42)
        pooled_pca = self.pca.fit_transform(pooled).astype(np.float64)

        print(f"Training GMM with {self.n_gaussians} Gaussians...")
        self.gmm = GaussianMixture(
            n_components=self.n_gaussians,
            covariance_type='diag',
            max_iter=100,
            n_init=1,
            reg_covar=1e-6,
            random_state=42
        )
        self.gmm.fit(pooled_pca)

        self.is_trained = True
        print(f"Fisher Vector model trained successfully")
        print(
            f"  - PCA variance explained: {self.pca.explained_variance_ratio_.sum():.2%}")
        print(f"  - GMM converged: {self.gmm.converged_}")

    def _prepare_image(self, image):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def encode_image(self, image):
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        img_gray = self._prepare_image(image)
        descriptors = self.sift_extractor.extract_multiscale(img_gray)

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(2 * self.n_gaussians * self.pca_dim)

        descriptors_pca = self.pca.transform(descriptors)
        posteriors = self.gmm.predict_proba(descriptors_pca)

        means = self.gmm.means_
        covariances = self.gmm.covariances_

        fv_mu = np.zeros((self.n_gaussians, self.pca_dim))
        fv_sigma = np.zeros((self.n_gaussians, self.pca_dim))

        for k in range(self.n_gaussians):
            diff = descriptors_pca - means[k]
            diff_norm = diff / np.sqrt(covariances[k] + 1e-7)
            weighted_diff = posteriors[:, k:k+1] * diff_norm
            fv_mu[k] = weighted_diff.sum(axis=0)
            fv_sigma[k] = (posteriors[:, k:k+1] *
                           (diff_norm**2 - 1)).sum(axis=0)

        fv = np.concatenate([fv_mu.flatten(), fv_sigma.flatten()])
        fv = np.sign(fv) * np.sqrt(np.abs(fv))
        fv = fv / (np.linalg.norm(fv) + 1e-7)

        return fv

    def encode_images(self, images):
        vectors = []
        for img in tqdm(images, desc="Encoding with Fisher Vectors"):
            fv = self.encode_image(img)
            vectors.append(fv)
        return np.array(vectors)

    def save(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        state = {
            'n_gaussians': self.n_gaussians,
            'pca_dim': self.pca_dim,
            'pca': self.pca,
            'gmm': self.gmm,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        extractor = cls(
            n_gaussians=state['n_gaussians'], pca_dim=state['pca_dim'])
        extractor.pca = state['pca']
        extractor.gmm = state['gmm']
        extractor.is_trained = state['is_trained']
        return extractor

    def get_feature_dimension(self):
        return 2 * self.n_gaussians * self.pca_dim
