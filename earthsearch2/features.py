"""DINOv2 feature extraction.

Single embedding strategy: CLS token concatenated with mean-pooled patch
tokens, L2-normalized. ResNet support was dropped — DINOv2 is what we use.

embed_batch runs N images through one forward pass — the major efficiency
win over per-image extraction during indexing.
"""

from __future__ import annotations

from typing import List, Literal, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import warnings 

warnings.filterwarnings( "ignore", message="xFormers is available.*")
warnings.filterwarnings( "ignore", message="xFormers is not available.*")

ModelName = Literal[
    "dinov2_vits14",
    "dinov2_vits14_reg",
    "dinov2_vitb14",
    "dinov2_vitb14_reg",
    "dinov2_vitl14",
    "dinov2_vitl14_reg",
    "dinov2_vitg14",
    "dinov2_vitg14_reg",
]

_BACKBONE_BASE_DIM = {
    "dinov2_vits14": 384,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14": 1536,
    "dinov2_vitg14_reg": 1536,
}


class FeatureExtractor:
    def __init__(
        self,
        model_type: ModelName = "dinov2_vits14_reg",
        device: str = None,
        input_size: int = 224,
    ) -> None:
        if model_type not in _BACKBONE_BASE_DIM:
            raise ValueError(f"Unsupported model_type: {model_type}")
        if input_size % 14 != 0:
            raise ValueError(f"input_size={input_size} must be a multiple of 14 for DINOv2")

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available() and "_reg" not in model_type:
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model_type = model_type
        self.input_size = input_size

        self.model = torch.hub.load("facebookresearch/dinov2", model_type, verbose=False)
        self.model = self.model.to(self.device).eval()

        # CLS + patch-mean concat → output dim is 2 * backbone dim
        self.embedding_dim = _BACKBONE_BASE_DIM[model_type] * 2

        self.transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _to_pil(src: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(src, str):
            return Image.open(src).convert("RGB")
        if isinstance(src, np.ndarray):
            return Image.fromarray(src).convert("RGB")
        return src.convert("RGB")

    def _forward(self, batch: torch.Tensor) -> np.ndarray:
        """Run the model on a (N, 3, H, W) tensor; return (N, 2D) unit-normalized embeddings.

        Uses bfloat16 autocast on CUDA — typically ~2x throughput vs fp32 on Ampere+
        and frees up enough memory to double the batch size. CPU and MPS stay in fp32.
        """
        batch = batch.to(self.device, non_blocking=True)
        use_amp = self.device == "cuda"
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    feats = self.model.forward_features(batch)
            else:
                feats = self.model.forward_features(batch)
        cls = feats["x_norm_clstoken"]
        patch_mean = feats["x_norm_patchtokens"].mean(dim=1)
        out = torch.cat([cls, patch_mean], dim=-1).float().cpu().numpy()
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out.astype(np.float32)

    # -- public API -----------------------------------------------------------

    def embed_batch(
        self, images: List[Union[str, np.ndarray, Image.Image]]
    ) -> np.ndarray:
        """Embed N images in a single forward pass. Returns (N, embedding_dim) float32."""
        tensors = [self.transform(self._to_pil(img)) for img in images]
        batch = torch.stack(tensors, dim=0)
        return self._forward(batch)

    def embed_tensor_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Embed a pre-transformed (N, 3, H, W) tensor batch. Returns (N, embedding_dim) float32.

        Lets DataLoader workers do the PIL→tensor→normalize work in parallel so the
        embedding loop on the main thread just hands batches to the GPU.
        """
        return self._forward(batch)

    def extract_embedding(self, src: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Embed a single image; returns 1D (embedding_dim,) float32."""
        return self.embed_batch([src])[0]

    def extract_query_embedding(
        self,
        src: Union[str, np.ndarray, Image.Image],
        rotations: Tuple[int, ...] = (0, 90, 180, 270),
    ) -> np.ndarray:
        """Average embeddings over 90° rotations of the query for orientation-invariance."""
        rotate_map = {
            0: None,
            90: Image.Transpose.ROTATE_90,
            180: Image.Transpose.ROTATE_180,
            270: Image.Transpose.ROTATE_270,
        }
        for r in rotations:
            if r not in rotate_map:
                raise ValueError(f"rotations must be in {{0, 90, 180, 270}}, got {r}")

        image = self._to_pil(src)
        rotated = [image if r == 0 else image.transpose(rotate_map[r]) for r in rotations]
        embeddings = self.embed_batch(rotated)
        avg = embeddings.mean(axis=0)
        avg /= np.linalg.norm(avg)
        return avg.astype(np.float32)
