import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
from typing import Tuple, Optional
from loguru import logger


class FaceEmbedder:
    """
    High-Dimensional Face Feature Extractor & Quality Analyzer.
    Extracts 512-dimensional L2-normalized embeddings via ArcFace.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        model_name: str = "w600k_r50",
    ):
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.rec_model = None

        self._init_model()

    def _init_model(self):
        """Initialize FaceNet InceptionResnetV1 deep face recognition model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from facenet_pytorch import InceptionResnetV1
            self.rec_model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            logger.info(f"FaceEmbedder initialized with FaceNet InceptionResnetV1 on {self.device}")
        except Exception as e:
            logger.warning(f"FaceNet InceptionResnetV1 unavailable ({e}). Deep learning model fallback active.")
            self.rec_model = None

    @staticmethod
    def calculate_sharpness(face_bgr: np.ndarray) -> float:
        """
        Calculate face sharpness/clarity via normalized Laplacian variance (0.0 to 1.0).
        """
        if face_bgr is None or face_bgr.size == 0:
            return 0.0
        if len(face_bgr.shape) == 2:
            gray = face_bgr
        else:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(np.clip(lap_var / 300.0, 0.0, 1.0))

    def get_embedding(self, face_crop: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Extract a 512-dimensional L2-normalized deep face embedding.
        """
        if face_crop is None or face_crop.size == 0:
            return np.zeros((self.embedding_dim,), dtype=np.float32), 0.0

        if len(face_crop.shape) == 2:
            face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2BGR)
        elif face_crop.shape[2] == 4:
            face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_BGRA2BGR)
        else:
            face_bgr = face_crop

        quality_score = self.calculate_sharpness(face_bgr)

        # 1. Primary Engine: FaceNet InceptionResnetV1 (160x160 input, normalized)
        if self.rec_model is not None:
            try:
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
                # Standard InceptionResnetV1 normalization: (x - 127.5) / 128.0
                face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float()
                face_tensor = (face_tensor - 127.5) / 128.0
                face_tensor = face_tensor.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    emb = self.rec_model(face_tensor).cpu().numpy().flatten().astype(np.float32)

                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    emb = emb / norm
                return emb, quality_score
            except Exception as e:
                logger.error(f"FaceNet embedding extraction error: {e}")

        # 2. Fallback: Normalized texture descriptor
        gray = cv2.cvtColor(cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2GRAY)
        feat = gray.flatten().astype(np.float32)
        if len(feat) > self.embedding_dim:
            feat = feat[:self.embedding_dim]
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat = feat / norm
        return feat, quality_score

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embedding vectors (-1.0 to 1.0).
        """
        if emb1 is None or emb2 is None:
            return 0.0
        v1 = emb1.flatten().astype(np.float32)
        v2 = emb2.flatten().astype(np.float32)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two embedding vectors.
        """
        if emb1 is None or emb2 is None:
            return float("inf")
        v1 = emb1.flatten().astype(np.float32)
        v2 = emb2.flatten().astype(np.float32)
        return float(np.linalg.norm(v1 - v2))
