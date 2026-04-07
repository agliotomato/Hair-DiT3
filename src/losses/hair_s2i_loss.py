"""
HairS2ILoss — SD3.5 Hair Sketch-to-Image 복합 손실 함수

L_total = L_flow + λ₁·L_bg + λ₂·L_structure + λ₃·L_perceptual

  L_flow (1.0):      Flow Matching v-prediction MSE
  L_bg   (3.0):      배경 영역 latent space L2 — 배경 보존 핵심
  L_structure (0.7): 스케치-엣지 정합 (Canny 기반)
  L_perceptual (0.2): VGG 지각 손실 (relu2_2, relu3_3)
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGPerceptualLoss(nn.Module):
    """torchvision VGG16 relu2_2, relu3_3 feature L1 손실"""

    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)

        features = vgg.features
        # relu2_2: features[0:9], relu3_3: features[0:16]
        self.slice1 = nn.Sequential(*list(features.children())[:9])   # relu2_2
        self.slice2 = nn.Sequential(*list(features.children())[:16])  # relu3_3

        for param in self.parameters():
            param.requires_grad_(False)

    def forward(
        self,
        pred: torch.Tensor,   # [B, 3, H, W] float32 in [-1, 1]
        target: torch.Tensor, # [B, 3, H, W]
        mask: Optional[torch.Tensor] = None,  # [B, 1, H, W] 또는 None
    ) -> torch.Tensor:
        # [-1, 1] → VGG 입력 범위 [0, 1]
        pred_n   = (pred.clamp(-1, 1) + 1) / 2
        target_n = (target.clamp(-1, 1) + 1) / 2

        if mask is not None:
            pred_n   = pred_n   * mask
            target_n = target_n * mask

        loss = torch.tensor(0.0, device=pred.device)
        for slicer in [self.slice1, self.slice2]:
            f_pred   = slicer(pred_n)
            f_target = slicer(target_n)
            loss = loss + F.l1_loss(f_pred, f_target)
        return loss


class CannyEdgeLoss(nn.Module):
    """kornia Canny 기반 스케치-엣지 정합 손실 (kornia 없으면 skip)"""

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
        pred_image: torch.Tensor,  # [B, 3, H, W]
        sketch:     torch.Tensor,  # [B, 1, H, W]
        matte:      torch.Tensor,  # [B, 1, H, W]
    ) -> torch.Tensor:
        if not self._has_kornia:
            return torch.tensor(0.0, device=pred_image.device)

        import kornia.filters
        # 예측 이미지 헤어 영역만
        pred_hair = pred_image * matte
        # Grayscale 변환
        pred_gray = pred_hair.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        # Canny edge 검출 (kornia Canny → magnitude 반환)
        _, edge_pred = kornia.filters.canny(pred_gray)
        edge_pred = edge_pred.float()

        # 스케치를 same resolution으로
        sketch_resized = F.interpolate(
            sketch * matte,
            size=edge_pred.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        return F.mse_loss(edge_pred, sketch_resized.clamp(0, 1))


class HairS2ILoss(nn.Module):
    """
    복합 손실 함수.

    L_total = L_flow + λ_bg·L_bg + λ_struct·L_structure + λ_percep·L_perceptual

    Args:
        lambda_bg:      배경 보존 손실 가중치 (기본 3.0, 수렴 후 1.5로 낮출 것)
        lambda_struct:  스케치 정합 손실 가중치 (기본 0.7)
        lambda_percep:  지각 손실 가중치 (기본 0.2)
        use_perceptual: VGG perceptual loss 사용 여부 (메모리 부족 시 False)
        use_structure:  Canny structure loss 사용 여부
    """

    def __init__(
        self,
        lambda_bg:      float = 3.0,
        lambda_struct:  float = 0.7,
        lambda_percep:  float = 0.2,
        use_perceptual: bool  = True,
        use_structure:  bool  = True,
    ):
        super().__init__()
        self.lambda_bg      = lambda_bg
        self.lambda_struct  = lambda_struct
        self.lambda_percep  = lambda_percep
        self.use_perceptual = use_perceptual
        self.use_structure  = use_structure

        if use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss()

        if use_structure:
            self.edge_loss = CannyEdgeLoss()

    def forward(
        self,
        noise_pred:    torch.Tensor,   # [B, 16, H/8, W/8]  DiT 예측
        noise_target:  torch.Tensor,   # [B, 16, H/8, W/8]  Flow Matching target
        pred_latent:   torch.Tensor,   # [B, 16, H/8, W/8]  예측 latent (디코딩 전)
        z_bg:          torch.Tensor,   # [B, 16, H/8, W/8]  원본 배경 latent
        matte_latent:  torch.Tensor,   # [B,  1, H/8, W/8]  latent 해상도 매트
        # 아래는 L_structure, L_perceptual 에만 사용 (optional)
        pred_image:    Optional[torch.Tensor] = None,  # [B, 3, H, W]
        target_image:  Optional[torch.Tensor] = None,  # [B, 3, H, W]
        sketch:        Optional[torch.Tensor] = None,  # [B, 1, H, W]
        matte_image:   Optional[torch.Tensor] = None,  # [B, 1, H, W] 원본 해상도
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss: scalar
            loss_dict: 각 손실 값 로깅용 딕셔너리
        """
        loss_dict = {}

        # ── L_flow: Flow Matching v-prediction MSE ──
        # target_flow = noise - target_latent (Flow Matching 공식)
        # Spatial weighting: 머리카락 영역(Matte)에 1.0, 그 외 배경에 0.1 가중치 부여
        weight = matte_latent + 0.1 * (1.0 - matte_latent)
        L_flow = (weight * (noise_pred - noise_target) ** 2).mean()
        loss_dict["loss_flow"] = L_flow.item()

        # ── L_bg: 배경 영역 latent L2 ──
        # 배경 영역(1-matte)에서 예측 latent와 원본 배경 latent 비교
        bg_mask = 1.0 - matte_latent
        L_bg = F.mse_loss(
            pred_latent * bg_mask,
            z_bg        * bg_mask,
        )
        loss_dict["loss_bg"] = L_bg.item()

        total = L_flow + self.lambda_bg * L_bg

        # ── L_structure: Canny edge 정합 ──
        if self.use_structure and pred_image is not None and sketch is not None and matte_image is not None:
            L_struct = self.edge_loss(pred_image, sketch, matte_image)
            loss_dict["loss_structure"] = L_struct.item()
            total = total + self.lambda_struct * L_struct

        # ── L_perceptual: VGG 지각 손실 ──
        if self.use_perceptual and pred_image is not None and target_image is not None and matte_image is not None:
            L_percep = self.perceptual_loss(pred_image, target_image, matte_image)
            loss_dict["loss_perceptual"] = L_percep.item()
            total = total + self.lambda_percep * L_percep

        loss_dict["loss_total"] = total.item()
        return total, loss_dict
