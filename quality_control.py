import numpy as np
from typing import List, Tuple


class QualityControl:
    """质量控制类，实现语义一致性过滤"""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def compute_centroid(self, vectors: List[np.ndarray]) -> np.ndarray:
        """计算质心向量"""
        return np.mean(vectors, axis=0)

    def compute_similarities(self, vectors: List[np.ndarray],
                             centroid: np.ndarray) -> np.ndarray:
        """计算每个向量与质心的余弦相似度"""
        similarities = []
        for vec in vectors:
            sim = np.dot(vec, centroid) / (np.linalg.norm(vec) * np.linalg.norm(centroid) + 1e-8)
            similarities.append(sim)
        return np.array(similarities)

    def filter_descriptions(self,
                            description_vectors: List[np.ndarray],
                            descriptions: List[str]) -> Tuple[List[str], List[np.ndarray], List[float]]:
        """过滤低质量描述"""
        if len(description_vectors) == 0:
            return [], [], []

        centroid = self.compute_centroid(description_vectors)
        similarities = self.compute_similarities(description_vectors, centroid)

        retained_indices = np.where(similarities >= self.threshold)[0]

        retained_descriptions = [descriptions[i] for i in retained_indices]
        retained_vectors = [description_vectors[i] for i in retained_indices]
        retained_similarities = [similarities[i] for i in retained_indices]

        return retained_descriptions, retained_vectors, retained_similarities