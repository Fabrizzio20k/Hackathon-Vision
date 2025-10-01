import faiss
import numpy as np
from enum import Enum
from typing import Optional, Tuple
import pickle
import os


class IndexType(Enum):
    FLAT_L2 = "flat_l2"
    FLAT_IP = "flat_ip"
    IVF_FLAT = "ivf_flat"
    IVF_PQ = "ivf_pq"
    HNSW = "hnsw"
    LSH = "lsh"
    PQ = "pq"


class FaissDatabase:
    def __init__(self, dimension: int, index_type: IndexType = IndexType.FLAT_L2):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.ids = []
        self.normalize = (index_type == IndexType.FLAT_IP)
        self._build_index()

    def _build_index(self):
        if self.index_type == IndexType.FLAT_L2:
            self.index = faiss.IndexFlatL2(self.dimension)

        elif self.index_type == IndexType.FLAT_IP:
            self.index = faiss.IndexFlatIP(self.dimension)

        elif self.index_type == IndexType.HNSW:
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16

        elif self.index_type == IndexType.LSH:
            nbits = self.dimension * 4
            self.index = faiss.IndexLSH(self.dimension, nbits)

        elif self.index_type == IndexType.IVF_FLAT:
            nlist = 100
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)

        elif self.index_type == IndexType.IVF_PQ:
            nlist = 100
            m = 8
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(
                quantizer, self.dimension, nlist, m, 8)

        elif self.index_type == IndexType.PQ:
            m = 8
            self.index = faiss.IndexPQ(self.dimension, m, 8)

    def train(self, vectors: np.ndarray):
        if not self.index.is_trained:
            print(f"Training {self.index_type.value} index...")
            vectors = vectors.astype('float32')
            self.index.train(vectors)
            print("Training complete.")

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        vectors = vectors.astype('float32')

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if self.normalize:
            faiss.normalize_L2(vectors)

        if not self.index.is_trained:
            self.train(vectors)

        n = len(vectors)
        if ids is None:
            ids = np.arange(len(self.ids), len(self.ids) + n)

        self.index.add(vectors)
        self.ids.extend(ids.tolist() if isinstance(ids, np.ndarray) else ids)

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        queries = queries.astype('float32')

        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        if self.normalize:
            faiss.normalize_L2(queries)

        distances, indices = self.index.search(queries, k)

        return distances, indices

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(
            filepath) else '.', exist_ok=True)

        faiss.write_index(self.index, f"{filepath}.index")

        metadata = {
            'dimension': self.dimension,
            'index_type': self.index_type.value,
            'ids': self.ids,
            'normalize': self.normalize
        }
        with open(f"{filepath}.meta", 'wb') as f:
            pickle.dump(metadata, f)

    @classmethod
    def load(cls, filepath: str):
        with open(f"{filepath}.meta", 'rb') as f:
            metadata = pickle.load(f)

        index_type = IndexType(metadata['index_type'])
        db = cls(metadata['dimension'], index_type)
        db.index = faiss.read_index(f"{filepath}.index")
        db.ids = metadata['ids']
        db.normalize = metadata.get(
            'normalize', index_type == IndexType.FLAT_IP)

        return db

    def size(self) -> int:
        return self.index.ntotal

    def reset(self):
        self.index.reset()
        self.ids = []
