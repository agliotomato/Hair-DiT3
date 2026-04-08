"""
TimestepAwareLatentCompositor

매 denoising 스텝에서 배경 보존 + 경계면 Artifact 방지.

SD3.5 Flow Matching 공식 (DDPM add_noise 사용 금지):
    z_t = (1 - sigma) * z_clean + sigma * noise
    sigma: 0(클린) → 1(최대노이즈)

타임스텝 기반 동적 blur:
    고sigma(노이즈 큼): blur_radius=0  → Hard Matte → 글로벌 구조 강제
    저sigma(클린에 가까움): blur_radius=max_blur → Soft Matte → 경계 자연화

파라미터 없음 (학습 대상 아님, purely functional).
"""
from typing import Optional

import torch
import torchvision.transforms.functional as TF


class TimestepAwareLatentCompositor:
    """
    denoising loop 내에서 배경 latent와 예측 latent를 sigma-aware하게 합성.

    사용법:
        compositor = TimestepAwareLatentCompositor()
        z = compositor(z_pred, z_bg, matte, sigma)
    """

    def __call__(
        self,
        z_pred:    torch.Tensor,            # [B, 16, H/8, W/8]  현재 스텝 예측 latent
        z_bg:      torch.Tensor,            # [B, 16, H/8, W/8]  원본 배경 clean latent
        matte:     torch.Tensor,            # [B,  1, H/8, W/8]  헤어=1.0, 배경=0.0
        sigma:     torch.Tensor,            # [B]  Flow Matching sigma (0=클린, 1=최대노이즈)
        noise:     torch.Tensor = None,     # [B, 16, H/8, W/8]  고정 배경 노이즈 (None이면 매 스텝 새로 생성)
        max_blur:  int = 15,
    ) -> torch.Tensor:
        """
        Args:
            z_pred: 현재 스텝 DiT 예측 latent
            z_bg:   원본 배경 clean latent (VAE encode 결과)
            matte:  latent 해상도로 다운샘플된 헤어 매트
            sigma:  현재 스텝 sigma 값 (FlowMatchEulerDiscreteScheduler.sigmas[t])
            max_blur: 저sigma(클린)에서 적용할 최대 Gaussian blur 반지름

        Returns:
            z_out: 합성된 latent [B, 16, H/8, W/8]
                   헤어 영역 = z_pred (Soft Matte 경계)
                   배경 영역 = z_bg_noised (원본 배경을 현재 noise 수준으로)
        """
        # ── 1. SD3.5 Flow Matching noising (DDPM add_noise 사용 금지) ──
        # noise를 외부에서 고정하면 배경 영역 trajectory가 일관됨 → 경계 아티팩트 감소
        if noise is None:
            noise = torch.randn_like(z_bg)
        s        = sigma.float().view(-1, 1, 1, 1).to(z_bg.device)
        z_bg_noised = (1.0 - s) * z_bg.float() + s * noise.float()
        z_bg_noised = z_bg_noised.to(z_bg.dtype)

        # ── 2. 동적 blur radius ──
        # sigma: 0(클린)→1(노이즈)
        # 저sigma(클린에 가까울수록) → blur 크게 → 경계 부드럽게
        # 고sigma(노이즈 클수록)    → blur 없음 → 구조 강제
        t_mean      = sigma.float().mean().item()
        blur_radius = int((1.0 - t_mean) * max_blur)

        # ── 3. Soft Matte 생성 ──
        if blur_radius > 0:
            # 공간 크기보다 padding이 크면 안 됨: kernel < 2 * spatial_dim
            H, W = matte.shape[-2], matte.shape[-1]
            max_kernel = min(H, W) - 1
            # kernel_size는 홀수, max_kernel을 넘지 않도록 clamp
            kernel_size = min(blur_radius * 2 + 1, max_kernel)
            if kernel_size % 2 == 0:
                kernel_size -= 1  # 홀수 보장
            if kernel_size > 1:
                soft_matte = TF.gaussian_blur(
                    matte,
                    kernel_size=[kernel_size, kernel_size],
                    sigma=kernel_size / 6.0,  # kernel_size 기반 sigma
                )
            else:
                soft_matte = matte
        else:
            soft_matte = matte

        # ── 4. 합성 ──
        # 헤어 영역 = z_pred (DiT 생성)
        # 배경 영역 = z_bg_noised (원본 배경, 현재 noise 수준 유지)
        z_out = z_pred * soft_matte + z_bg_noised * (1.0 - soft_matte)

        return z_out
