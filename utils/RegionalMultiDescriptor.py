import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class RegionalMultiDescriptor:
    def __init__(self, grid_size=6, hog_orientations=9, lbp_radius=1, lbp_points=8, color_bins=6):
        # Grid 6x6 = 36 regiones de 16x16 pixels
        self.grid_size = grid_size
        self.hog_orientations = hog_orientations
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.color_bins = color_bins  # Reducir bins para evitar overfitting
        self.feature_dim = self._calculate_dimension()

    def _calculate_dimension(self):
        n_regions = self.grid_size * self.grid_size
        hog_dim = self.hog_orientations  # Solo 9, no multiplicar por 4
        lbp_dim = self.lbp_points + 2
        color_dim = self.color_bins * 3
        return n_regions * (hog_dim + lbp_dim + color_dim)

    def _extract_hog_region(self, region):
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if len(region.shape) == 3 else region
        
        # HOG simple sin bloques
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
        
        mag = np.sqrt(gx**2 + gy**2)
        ang = np.arctan2(gy, gx) * 180 / np.pi
        ang[ang < 0] += 180
        
        hist = np.zeros(self.hog_orientations)
        bin_width = 180.0 / self.hog_orientations
        
        for i in range(region.shape[0]):
            for j in range(region.shape[1]):
                bin_idx = min(int(ang[i, j] / bin_width), self.hog_orientations - 1)
                hist[bin_idx] += mag[i, j]
        
        return hist / (np.sum(hist) + 1e-6)

    def _extract_lbp_region(self, region):
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if len(region.shape) == 3 else region
        lbp = local_binary_pattern(gray, self.lbp_points, self.lbp_radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=self.lbp_points + 2, range=(0, self.lbp_points + 2))
        return hist.astype(np.float32) / (hist.sum() + 1e-7)

    def _extract_color_histogram_region(self, region):
        if len(region.shape) == 2:
            return np.zeros(self.color_bins * 3)
        
        # Usar HSV en lugar de LAB para mejor discriminación de color
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        features = []
        
        # H: [0, 180], S: [0, 255], V: [0, 255]
        hist_h, _ = np.histogram(hsv[:,:,0], bins=self.color_bins, range=(0, 180))
        hist_s, _ = np.histogram(hsv[:,:,1], bins=self.color_bins, range=(0, 256))
        hist_v, _ = np.histogram(hsv[:,:,2], bins=self.color_bins, range=(0, 256))
        
        hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
        return hist / (hist.sum() + 1e-7)

    def extract(self, image):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        h, w = image.shape[:2]
        region_h = h // self.grid_size
        region_w = w // self.grid_size
        
        all_features = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1 = i * region_h
                y2 = min((i + 1) * region_h, h)
                x1 = j * region_w
                x2 = min((j + 1) * region_w, w)
                
                region = image[y1:y2, x1:x2]
                
                # Extraer y concatenar sin pesos
                hog = self._extract_hog_region(region)
                lbp = self._extract_lbp_region(region)
                color = self._extract_color_histogram_region(region)
                
                all_features.extend(hog)
                all_features.extend(lbp)
                all_features.extend(color)
        
        # Normalización L2
        features = np.array(all_features, dtype=np.float32)
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features

    def get_feature_dimension(self):
        return self.feature_dim