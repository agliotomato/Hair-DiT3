"""
HairS2ILoss — SD3.5 Hair Sketch-to-Image 복합 손실 함수

L_total = L_flow + λ_bg·L_bg + λ_color·L_color + λ_lpips·L_lpips + λ_edge·L_edge

  L_flow  (1.0):  Flow Matching masked MSE — 헤어=1.0, 배경=0.1
  L_bg    (0.5):  배경 latent L2 보존
  L_color (1.5):  sketch_latent ≈ pred_z0 헤어 영역 latent MSE — 색 following 핵심
  L_lpips (0.1):  LPIPS 지각 손실 — 헤어 영역만
  L_edge  (0.05): Sobel edge vs sketch 정합 — sketch stroke 있는데 edge 없으면 패널티

Phase 1 (pretrain):  L_flow + L_bg + L_color  →  step 30% 이후 L_lpips 추가
Phase 2 (finetune):  L_flow + L_bg + L_color + L_lpips + L_edge
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LPIPSLoss(nn.Module):
    """LPIPS (Learned Perceptual Image Patch Similarity) 손실."""

    def __init__(self):
        super().__init__()
        self._lpips = None
        try:
            import lpips
            self._lpips = lpips.LPIPS(net="vgg")
            for p in self._lpips.parameters():
                p.requires_grad_(False)
        except ImportError:
            pass

    def forward(
        self,
        pred:   torch.Tensor,           # [B, 3, H, W] in [-1, 1]
        target: torch.Tensor,           # [B, 3, H, W] in [-1, 1]
        mask:   Optional[torch.Tensor] = None,  # [B, 1, H, W]
    ) -> torch.Tensor:
        if self._lpips is None:
            return torch.tensor(0.0, device=pred.device)

        if mask is not None:
            pred   = pred   * mask
            target = target * mask

        # lpips는 [-1, 1] 입력 그대로 받음
        return self._lpips(pred, target).mean()


class SobelEdgeLoss(nn.Module):
    """
    Sobel edge 기반 스케치 정합 손실.

    sketch stroke가 있는 영역에서 생성 이미지에 edge가 없으면 패널티.
    단방향 패널티 → 과도한 edge는 허용, 누락된 edge만 패널티.

    loss = mean(sketch_mask * (1 - edge_mag))
    """

    def __init__(self):
        super().__init__()
        self._has_kornia = False
        try:
            import kornia.filters
            self._has_kornia = True
        except ImportError:
            pass

    def forward(
        self,
        pred_image: torch.Tensor,  # [B, 3, H, W] in [-1, 1]
        sketch:     torch.Tensor,  # [B, 3, H, W] in [-1, 1] (RGB 스케치)
        matte:      torch.Tensor,  # [B, 1, H, W] in [0, 1]
    ) -> torch.Tensor:
        if not self._has_kornia:
            return torch.tensor(0.0, device=pred_image.device)

        import kornia.filters

        # 헤어 영역 grayscale
        pred_gray = (pred_image * matte).mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Sobel gradient → edge magnitude
        grads    = kornia.filters.spatial_gradient(pred_gray)  # [B, 1, 2, H, W]
        edge_mag = grads.norm(dim=2).clamp(0, 1)               # [B, 1, H, W]

        # sketch stroke mask: sketch [-1,1] → [0,1], max channel > 0.1
        sketch_01   = (sketch + 1.0) / 2.0                           # [B, 3, H, W]
        sketch_mask = (sketch_01.max(dim=1, keepdim=True).values > 0.1).float()

        # 패널티: stroke 있는데 edge 없는 영역
        return (sketch_mask * (1.0 - edge_mag)).mean()


class HairS2ILoss(nn.Module):
    """
    복합 손실 함수.

    Args:
        phase:            1 (pretrain) 또는 2 (finetune)
        lambda_bg:        배경 보존 가중치 (기본 3.0)
        lambda_lpips:     LPIPS 가중치 (기본 0.1)
        lambda_edge:      Sobel edge 가중치 (기본 0.05)
        lpips_warmup_ratio: Phase 1에서 LPIPS 시작 비율 (기본 0.3 = 30%)
    """

    def __init__(
        self,
        phase:              int   = 1,
        lambda_bg:          float = 0.5,
        lambda_color:       float = 1.5,
        lambda_lpips:       float = 0.1,
        lambda_edge:        float = 0.05,
        lpips_warmup_ratio: float = 0.3,
    ):
        super().__init__()
        self.phase              = phase
        self.lambda_bg          = lambda_bg
        self.lambda_color       = lambda_color
        self.lambda_lpips       = lambda_lpips
        self.lambda_edge        = lambda_edge
        self.lpips_warmup_ratio = lpips_warmup_ratio

        self.lpips_loss = LPIPSLoss()
        self.edge_loss  = SobelEdgeLoss()

    def forward(
        self,
        noise_pred:    torch.Tensor,            # [B, 16, H/8, W/8]
        noise_target:  torch.Tensor,            # [B, 16, H/8, W/8]
        pred_latent:   torch.Tensor,            # [B, 16, H/8, W/8]
        z_bg:          torch.Tensor,            # [B, 16, H/8, W/8]
        matte_latent:  torch.Tensor,            # [B,  1, H/8, W/8]
        sketch_latent: Optional[torch.Tensor] = None,  # [B, 16, H/8, W/8]
        pred_image:    Optional[torch.Tensor] = None,  # [B, 3, H, W]
        target_image:  Optional[torch.Tensor] = None,  # [B, 3, H, W]
        sketch:        Optional[torch.Tensor] = None,  # [B, 3, H, W]
        matte_image:   Optional[torch.Tensor] = None,  # [B, 1, H, W]
        global_step:   int = 0,
        total_steps:   int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}

        # ── L_flow: Flow Matching masked MSE ──
        weight = matte_latent + 0.1 * (1.0 - matte_latent)
        L_flow = (weight * (noise_pred - noise_target) ** 2).mean()
        loss_dict["loss_flow"] = L_flow.item()

        # ── L_bg: 배경 latent L2 ──
        bg_mask = 1.0 - matte_latent
        L_bg = F.mse_loss(pred_latent * bg_mask, z_bg * bg_mask)
        loss_dict["loss_bg"] = L_bg.item()

        total = L_flow + self.lambda_bg * L_bg

        # ── L_color: sketch_latent ≈ pred_latent (헤어 영역 latent MSE) ──
        if sketch_latent is not None and self.lambda_color > 0:
            L_color = F.mse_loss(
                pred_latent * matte_latent,
                sketch_latent.float() * matte_latent,
            )
            loss_dict["loss_color"] = L_color.item()
            total = total + self.lambda_color * L_color

        # ── LPIPS 활성화 조건 ──
        # Phase 1: total_steps의 30% 이후에 활성
        # Phase 2: 항상 활성
        lpips_active = (
            self.phase == 2
            or (global_step / max(total_steps, 1)) >= self.lpips_warmup_ratio
        )

        if lpips_active and pred_image is not None and target_image is not None and matte_image is not None:
            L_lpips = self.lpips_loss(pred_image, target_image, matte_image)
            loss_dict["loss_lpips"] = L_lpips.item()
            total = total + self.lambda_lpips * L_lpips

        # ── L_edge: Sobel edge 정합 (Phase 2만) ──
        if self.phase == 2 and pred_image is not None and sketch is not None and matte_image is not None:
            L_edge = self.edge_loss(pred_image, sketch, matte_image)
            loss_dict["loss_edge"] = L_edge.item()
            total = total + self.lambda_edge * L_edge

        loss_dict["loss_total"] = total.item()
        return total, loss_dict
