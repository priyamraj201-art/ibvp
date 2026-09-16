"""
Person Re-Identification (Person Re-ID) Feature Extractor.
Extracts 512-dimensional appearance embeddings from full-body person bounding box crops
using the ultra-lightweight OSNet-x0.25 deep neural network.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from loguru import logger

from .osnet import osnet_x0_25


class PersonReIDExtractor:
    """
    Ultra-lightweight Deep Appearance Feature Extractor for Full-Body Person Re-ID.
    - Model: OSNet-x0.25 (~2.2 MB, ~4ms latency)
    - Input: BGR person bounding box crop
    - Output: L2-normalized 512-dimensional embedding vector
    - Supports CUDA FP16 and CPU execution
    """

    def __init__(
        self,
        device: Optional[str] = None,
        fp16: bool = True,
        weights_dir: str = "pretrained",
        embedding_dim: int = 512,
    ):
        self.embedding_dim = embedding_dim
        self.fp16 = fp16

        # Determine target device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.device_name = (
            f"GPU ({torch.cuda.get_device_name(0)})"
            if self.device.type == "cuda"
            else "CPU"
        )

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.model = None
        self._init_model(weights_dir)

    def _init_model(self, weights_dir: str):
        """Build and configure OSNet-x0.25 model."""
        try:
            model = osnet_x0_25(
                num_classes=1000,
                pretrained=True,
                weights_dir=weights_dir,
            )
            model.eval()
            model.to(self.device)

            # Enable FP16 for CUDA
            if self.fp16 and self.device.type == "cuda":
                model.half()

            self.model = model
            logger.info(
                f"[PersonReIDExtractor] OSNet-x0.25 ready on {self.device_name} (FP16={self.fp16 and self.device.type == 'cuda'})"
            )
        except Exception as e:
            logger.warning(
                f"[PersonReIDExtractor] Failed to build OSNet ({e}). Fallback descriptor active."
            )
            self.model = None

    def preprocess(self, crop: np.ndarray) -> Optional[torch.Tensor]:
        """
        Preprocess single BGR crop:
        - Resize to (256, 128) [height, width]
        - Convert BGR to RGB
        - Normalize to [0, 1] and standard ImageNet mean/std
        - Convert to PyTorch Tensor [1, 3, 256, 128]
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        # BGR -> RGB
        if len(crop.shape) == 2:
            rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[2] == 4:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Standard Re-ID aspect ratio: 256 height x 128 width
        resized = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - self.mean) / self.std

        # HWC -> CHW -> NCHW
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(self.device)

        if self.fp16 and self.device.type == "cuda":
            tensor = tensor.half()

        return tensor

    def extract_feature(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract a single 512-dimensional L2-normalized embedding for a person crop.
        """
        if crop is None or crop.size == 0:
            return np.zeros((self.embedding_dim,), dtype=np.float32)

        if self.model is not None:
            try:
                tensor = self.preprocess(crop)
                if tensor is not None:
                    with torch.no_grad():
                        features = self.model(tensor)
                        # L2 normalization: feat / ||feat||_2
                        features = F.normalize(features, p=2, dim=1)
                        emb = features.cpu().float().numpy().flatten().astype(np.float32)
                        return emb
            except Exception as e:
                logger.error(f"[PersonReIDExtractor] Extraction error: {e}")

        # Fallback: Multi-scale spatial color histogram (HSV + texture)
        return self._extract_fallback_descriptor(crop)

    def extract_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract features for a batch of crops in a single forward pass.
        """
        if not crops:
            return []

        tensors = []
        valid_indices = []

        for idx, crop in enumerate(crops):
            t = self.preprocess(crop)
            if t is not None:
                tensors.append(t)
                valid_indices.append(idx)

        results = [np.zeros((self.embedding_dim,), dtype=np.float32) for _ in crops]

        if not tensors or self.model is None:
            for idx in range(len(crops)):
                results[idx] = self._extract_fallback_descriptor(crops[idx])
            return results

        try:
            batch_tensor = torch.cat(tensors, dim=0)
            with torch.no_grad():
                features = self.model(batch_tensor)
                features = F.normalize(features, p=2, dim=1)
                embs = features.cpu().float().numpy()

            for valid_idx, emb in zip(valid_indices, embs):
                results[valid_idx] = emb.astype(np.float32)
        except Exception as e:
            logger.error(f"[PersonReIDExtractor] Batch extraction error: {e}")
            for idx in range(len(crops)):
                results[idx] = self._extract_fallback_descriptor(crops[idx])

        return results

    def _extract_fallback_descriptor(self, crop: np.ndarray) -> np.ndarray:
        """
        Deterministic spatial color-texture descriptor (512-D) when deep model is unavailable.
        Uses 3-stripe horizontal partitions (head/torso/legs) in HSV space.
        """
        if crop is None or crop.size == 0:
            return np.zeros((self.embedding_dim,), dtype=np.float32)

        try:
            resized = cv2.resize(crop, (64, 128))
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            h, w = hsv.shape[:2]

            # 3 vertical stripes: upper body, mid body, lower body
            stripes = [
                hsv[0 : h // 3, :],
                hsv[h // 3 : 2 * h // 3, :],
                hsv[2 * h // 3 :, :],
            ]

            hist_parts = []
            for stripe in stripes:
                # 8 hue bins, 4 sat bins, 4 val bins = 128 bins per stripe
                hist = cv2.calcHist([stripe], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
                hist = hist.flatten()
                norm = np.linalg.norm(hist)
                if norm > 1e-6:
                    hist = hist / norm
                hist_parts.append(hist)

            # Concatenate 3 stripes: 3 * 128 = 384 bins
            full_desc = np.concatenate(hist_parts)
            # Pad to 512-dim
            padded = np.zeros((self.embedding_dim,), dtype=np.float32)
            padded[: len(full_desc)] = full_desc
            norm = np.linalg.norm(padded)
            if norm > 1e-6:
                padded = padded / norm
            return padded
        except Exception:
            return np.zeros((self.embedding_dim,), dtype=np.float32)

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two 512-dimensional embeddings.
        Since embeddings are L2-normalized, cosine similarity is simply the dot product.
        """
        if emb1 is None or emb2 is None:
            return 0.0
        e1 = emb1.flatten()
        e2 = emb2.flatten()
        if len(e1) != len(e2) or len(e1) == 0:
            return 0.0

        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0

        sim = float(np.dot(e1, e2) / (n1 * n2))
        return float(np.clip(sim, -1.0, 1.0))
