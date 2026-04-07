"""
Augmentation pipeline for hair-DiT training.
Ported from hair-dit project with adjustments for hair-dit3 architecture.
"""

import random
import torch
import torch.nn.functional as F
import kornia.filters as KF
import kornia.morphology as KM

def soft_composite(img: torch.Tensor, matte: torch.Tensor) -> torch.Tensor:
    """
    Compute hair region image: img * matte.
    Args:
        img:   (C, H, W) or (B, C, H, W)
        matte: (1, H, W) or (B, 1, H, W), values in [0, 1]
    Returns:
        hair_region: same shape as img
    """
    return img * matte

class StrokeColorSampler:
    """
    Per-stroke color sampling from actual target hair pixels.
    Implementing the 33% Random / 67% Mean sampling logic.
    """

    def __init__(self, p: float = 1.0, min_pixels: int = 10, quantize_bits: int = 5):
        self.p = p
        self.min_pixels = min_pixels
        self.shift = 8 - quantize_bits

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        sketch = sample["sketch"]  # (3, H, W) float32 [-1, 1]
        target = sample["target"]  # (3, H, W) float32 [-1, 1] (already masked by matte)

        sketch_aug = self._resample_colors(sketch, target)
        sample["sketch"] = sketch_aug
        return sample

    def _resample_colors(self, sketch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1] for quantization logic
        sketch_norm = (sketch + 1.0) / 2.0
        sketch_u8 = (sketch_norm.clamp(0, 1) * 255).byte()
        sketch_q = (sketch_u8 >> self.shift) << self.shift

        flat_q = sketch_q.view(3, -1).T
        unique_colors = torch.unique(flat_q, dim=0)

        out = sketch.clone()

        for color in unique_colors:
            r, g, b = color.tolist()

            # Background or very dark colors (non-hair stroke) skipped
            # Since normalized back to [0, 255], 0 is roughly -1.0
            if r == 0 and g == 0 and b == 0:
                continue

            mask = (
                (sketch_q[0] == r) &
                (sketch_q[1] == g) &
                (sketch_q[2] == b)
            )

            hair_pixels = target[:, mask]  # (3, N) in [-1, 1]

            # Matte check: target is usually 0 (or -1 in [-1,1] range) in non-hair areas.
            # However, `target` passed from dataset should already be target * matte.
            # If target is [-1, 1], non-hair area is -1.0.
            valid = (hair_pixels.sum(dim=0) > -2.9)  # (N,)  Sum of -1, -1, -1 is -3.0
            
            if valid.sum() < self.min_pixels:
                continue

            hair_pixels_valid = hair_pixels[:, valid]

            # 33% random pixel, 67% mean pixel
            if random.randint(0, 5) < 2:
                idx = random.randint(0, hair_pixels_valid.shape[1] - 1)
                sampled_color = hair_pixels_valid[:, idx]
            else:
                sampled_color = hair_pixels_valid.mean(dim=1)

            out[:, mask] = sampled_color.unsqueeze(1)

        return out

class ThicknessJitter:
    """Randomly dilate sketch strokes by 0-2 pixels."""
    def __init__(self, p: float = 0.5, max_kernel: int = 3):
        self.p = p
        self.max_kernel = max_kernel

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        sketch = sample["sketch"].unsqueeze(0)  # (1, 3, H, W)
        k = random.choice([k for k in range(3, self.max_kernel + 1, 2)])
        kernel = torch.ones(k, k, device=sketch.device)
        sketch_dilated = KM.dilation(sketch, kernel).squeeze(0)
        sample["sketch"] = sketch_dilated.clamp(-1, 1)
        return sample

class MatteBoundaryPerturbation:
    """Apply elastic deformation to the matte to make the model robust to imperfect masks."""
    def __init__(self, p: float = 0.3, amplitude: float = 4.0, sigma: float = 10.0):
        self.p = p
        self.amplitude = amplitude
        self.sigma = sigma

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        matte = sample["matte"].unsqueeze(0)   # (1, 1, H, W)
        img   = sample["target"].unsqueeze(0)  # (1, 3, H, W) - we warp the target accordingly?
        # Actually, in hair-dit3, we should warp target and matte together to stay consistent.

        H, W = matte.shape[-2], matte.shape[-1]
        noise = torch.randn(1, 2, H, W, device=matte.device) * self.amplitude
        kernel_size = int(6 * self.sigma / H * H) | 1
        kernel_size = max(kernel_size, 3)
        noise = KF.gaussian_blur2d(noise, (kernel_size, kernel_size), (self.sigma, self.sigma))

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=matte.device),
            torch.linspace(-1, 1, W, device=matte.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        disp = noise.permute(0, 2, 3, 1)
        disp[..., 0] /= (W / 2)
        disp[..., 1] /= (H / 2)
        grid_warped = (grid + disp).clamp(-1, 1)

        matte_warped = F.grid_sample(matte.float(), grid_warped, mode="bilinear", align_corners=True)
        matte_warped = matte_warped.squeeze(0).clamp(0, 1)

        # Re-composite target image with warped matte
        # In training, 'target' is background + hair. 
        # But for 'L_flow' we care about the hair area.
        # Actually, warping the matte and re-compositing target ensures the boundary loss is consistent.
        # However, hair-dit3 usually has full 'target'. 
        # Let's just update the matte.
        sample["matte"] = matte_warped
        return sample

class ComposeAug:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample

def build_augmentation_pipeline(phase: str = "train") -> ComposeAug:
    return ComposeAug([
        StrokeColorSampler(p=1.0),
        ThicknessJitter(p=0.5),
        MatteBoundaryPerturbation(p=0.3),
    ])
