import cv2
import numpy as np


class HOGExtractor:
    def __init__(self, cells_per_block=(2, 2), pixels_per_cell=(8, 8), orientations=8):
        self.cells_per_block = cells_per_block
        self.pixels_per_cell = pixels_per_cell
        self.orientations = orientations

        self.feature_dim = self._calculate_dimension()

    def _calculate_dimension(self):
        cells_x = 96 // self.pixels_per_cell[0]
        cells_y = 96 // self.pixels_per_cell[1]
        blocks_x = cells_x - self.cells_per_block[0] + 1
        blocks_y = cells_y - self.cells_per_block[1] + 1
        return blocks_x * blocks_y * self.cells_per_block[0] * self.cells_per_block[1] * self.orientations

    def extract(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        hog = cv2.HOGDescriptor(
            _winSize=(96, 96),
            _blockSize=(self.pixels_per_cell[0] * self.cells_per_block[0],
                        self.pixels_per_cell[1] * self.cells_per_block[1]),
            _blockStride=(self.pixels_per_cell[0], self.pixels_per_cell[1]),
            _cellSize=self.pixels_per_cell,
            _nbins=self.orientations
        )

        features = hog.compute(gray)
        return features.flatten()

    def get_feature_dimension(self) -> int:
        return self.feature_dim
