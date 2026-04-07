"""전처리 유틸리티"""
from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def preprocess_image(
    img: Image.Image,
    size: Tuple[int, int] = (512, 512),
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """PIL Image → [1, 3, H, W] float32 in [-1, 1]"""
    img = img.convert("RGB").resize(size, Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def preprocess_sketch(
    img: Image.Image,
    size: Tuple[int, int] = (512, 512),
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """헤어 스케치 PIL → [1, 3, H, W] float32 in [-1, 1]"""
    img = img.convert("RGB").resize(size, Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def preprocess_matte(
    img: Image.Image,
    size: Tuple[int, int] = (512, 512),
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """헤어 매트 PIL → [1, 1, H, W] float32 in [0, 1]"""
    img = img.convert("L").resize(size, Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def postprocess_image(tensor: torch.Tensor) -> Image.Image:
    """[1, 3, H, W] float32 in [-1, 1] → PIL Image"""
    tensor = tensor.squeeze(0).float().clamp(-1, 1)
    arr = ((tensor.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(arr)


def matte_to_latent(
    matte: torch.Tensor,
    latent_size: Tuple[int, int],
) -> torch.Tensor:
    """[B, 1, H, W] 매트를 latent 해상도로 다운샘플 (mode='area')"""
    return F.interpolate(matte, size=latent_size, mode="area")
