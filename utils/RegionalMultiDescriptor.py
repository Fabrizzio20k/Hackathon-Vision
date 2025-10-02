import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class RegionalMultiDescriptor:
    def __init__(self, grid_size=6, hog_orientations=9, lbp_radius=1,
                 lbp_points=8, color_bins=6,
                 levels=None, lbp_configs=None):

        self.levels = levels if levels is not None else [2, 4, 6]
        self.lbp_configs = lbp_configs if lbp_configs is not None else [
            (1, 8), (2, 16)]
        self.hog_orientations = hog_orientations
        self.color_bins = color_bins
        self.feature_dim = self._calculate_dimension()

    def _calculate_dimension(self):
        hog_dim = self.hog_orientations
        lbp_dim = sum(n_points + 2 for _, n_points in self.lbp_configs)
        color_dim = self.color_bins * 3
        region_dim = hog_dim + lbp_dim + color_dim
        total = sum(level * level * region_dim for level in self.levels)
        return total

    def _extract_hog_region(self, region):
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if len(
            region.shape) == 3 else region

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)

        mag = np.sqrt(gx**2 + gy**2)
        ang = np.arctan2(gy, gx) * 180 / np.pi
        ang[ang < 0] += 180

        hist = np.zeros(self.hog_orientations)
        bin_width = 180.0 / self.hog_orientations

        for i in range(region.shape[0]):
            for j in range(region.shape[1]):
                bin_idx = min(int(ang[i, j] / bin_width),
                              self.hog_orientations - 1)
                hist[bin_idx] += mag[i, j]

        return hist / (np.sum(hist) + 1e-6)

    def _extract_lbp_region_multiscale(self, region):
        """LBP multi-escala"""
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if len(
            region.shape) == 3 else region

        all_features = []

        for radius, n_points in self.lbp_configs:
            lbp = local_binary_pattern(
                gray, n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(),
                                   bins=n_points + 2,
                                   range=(0, n_points + 2))
            hist_norm = hist.astype(np.float32) / (hist.sum() + 1e-7)
            all_features.extend(hist_norm)

        return np.array(all_features)

    def _extract_color_histogram_region(self, region):
        if len(region.shape) == 2:
            return np.zeros(self.color_bins * 3)

        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)

        hist_h, _ = np.histogram(
            hsv[:, :, 0], bins=self.color_bins, range=(0, 180))
        hist_s, _ = np.histogram(
            hsv[:, :, 1], bins=self.color_bins, range=(0, 256))
        hist_v, _ = np.histogram(
            hsv[:, :, 2], bins=self.color_bins, range=(0, 256))

        hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
        return hist / (hist.sum() + 1e-7)

    def _extract_level(self, image, grid_size):
        """Extrae features de un nivel de la pirámide"""
        h, w = image.shape[:2]
        region_h = h // grid_size
        region_w = w // grid_size

        level_features = []

        for i in range(grid_size):
            for j in range(grid_size):
                y1 = i * region_h
                y2 = min((i + 1) * region_h, h)
                x1 = j * region_w
                x2 = min((j + 1) * region_w, w)

                region = image[y1:y2, x1:x2]

                hog = self._extract_hog_region(region)
                lbp = self._extract_lbp_region_multiscale(region)
                color = self._extract_color_histogram_region(region)

                level_features.extend(hog)
                level_features.extend(lbp)
                level_features.extend(color)

        return np.array(level_features)

    def extract(self, image):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        all_features = []

        for idx, level in enumerate(self.levels):
            level_feats = self._extract_level(image, level)

            weight = 2 ** (-(len(self.levels) - idx - 1))

            all_features.extend(level_feats * weight)

        features = np.array(all_features, dtype=np.float32)
        features = np.sign(features) * np.sqrt(np.abs(features))

        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    def get_feature_dimension(self):
        return self.feature_dim
