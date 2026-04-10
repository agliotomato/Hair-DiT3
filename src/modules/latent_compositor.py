"""
TimestepAwareLatentCompositor

매 denoising 스텝에서 배경 보존 + 경계면 Artifact 방지.

SD3.5 Flow Matching 공식 (DDPM add_noise 사용 금지):
    z_t = (1 - sigma) * z_clean + sigma * noise
    sigma: 0(클린) → 1(최대노이즈)

타임스텝 기반 동적 blur:
    고sigma(노이즈 큼): blur_radius=0  → Hard Matte → 글로벌 구조 강제
    저sigma(클린에 가까움): blur_radius=max_blur → Soft Matte → 경계 자연화

noise_mode:
    "fixed"  : 외부에서 고정된 noise 재사용 → 배경 trajectory 일관성 (기본값)
    "random" : 매 스텝 새 noise 생성 → artifact 패턴 고착 방지, 단 일관성 감소

blur_schedule:
    "linear" : blur = (1 - sigma) * max_blur  (기본값)
    "cosine" : blur = (1 - cos((1 - sigma) * π)) / 2 * max_blur
               → 초반(고sigma) hard 유지, 후반(저sigma) soft 전환이 더 자연스러움

파라미터 없음 (학습 대상 아님, purely functional).
"""
from typing import Optional, Literal
import math

import torch
import torchvision.transforms.functional as TF


class TimestepAwareLatentCompositor:
    """
    denoising loop 내에서 배경 latent와 예측 latent를 sigma-aware하게 합성.

    사용법:
        compositor = TimestepAwareLatentCompositor()
        z = compositor(z_pred, z_bg, matte, sigma,
                       noise_mode="fixed", blur_schedule="linear")
    """

    def __call__(
        self,
        z_pred:        torch.Tensor,            # [B, 16, H/8, W/8]  현재 스텝 예측 latent
        z_bg:          torch.Tensor,            # [B, 16, H/8, W/8]  원본 배경 clean latent
        matte:         torch.Tensor,            # [B,  1, H/8, W/8]  헤어=1.0, 배경=0.0
        sigma:         torch.Tensor,            # [B]  Flow Matching sigma (0=클린, 1=최대노이즈)
        noise:         Optional[torch.Tensor] = None,  # [B, 16, H/8, W/8] 고정 배경 노이즈
        max_blur:      int = 15,
        noise_mode:    Literal["fixed", "random"] = "fixed",
        blur_schedule: Literal["linear", "cosine"] = "linear",
    ) -> torch.Tensor:
        """
        Args:
            z_pred:        현재 스텝 DiT 예측 latent
            z_bg:          원본 배경 clean latent (VAE encode 결과)
            matte:         latent 해상도로 다운샘플된 헤어 매트
            sigma:         현재 스텝 sigma 값 (FlowMatchEulerDiscreteScheduler.sigmas[t])
            noise:         고정 배경 노이즈. noise_mode="random"이면 무시됨
            max_blur:      저sigma(클린)에서 적용할 최대 Gaussian blur 반지름
            noise_mode:    "fixed" (trajectory 일관성) | "random" (매 스텝 새 noise)
            blur_schedule: "linear" | "cosine"

        Returns:
            z_out: 합성된 latent [B, 16, H/8, W/8]
                   헤어 영역 = z_pred (Soft Matte 경계)
                   배경 영역 = z_bg_noised (원본 배경을 현재 noise 수준으로)
        """
        # ── 1. noise 결정 ──
        if noise_mode == "random" or noise is None:
            bg_noise = torch.randn_like(z_bg)
        else:
            # noise_mode == "fixed": 외부에서 전달된 noise 재사용
            bg_noise = noise

        # ── 2. SD3.5 Flow Matching noising ──
        s = sigma.float().view(-1, 1, 1, 1).to(z_bg.device)
        z_bg_noised = (1.0 - s) * z_bg.float() + s * bg_noise.float()
        z_bg_noised = z_bg_noised.to(z_bg.dtype)

        # ── 3. blur radius 계산 ──
        t_mean = sigma.float().mean().item()  # [0, 1], 0=클린, 1=최대노이즈

        if blur_schedule == "cosine":
            # (1 - cos((1 - sigma) * π)) / 2
            # sigma=1(노이즈): (1-cos(0))/2 = 0   → blur 없음
            # sigma=0(클린):   (1-cos(π))/2 = 1   → max_blur
            weight = (1.0 - math.cos((1.0 - t_mean) * math.pi)) / 2.0
        else:
            # linear (기본값)
            weight = 1.0 - t_mean

        blur_radius = int(weight * max_blur)

        # ── 4. Soft Matte 생성 ──
        if blur_radius > 0:
            H, W = matte.shape[-2], matte.shape[-1]
            max_kernel = min(H, W) - 1
            kernel_size = min(blur_radius * 2 + 1, max_kernel)
            if kernel_size % 2 == 0:
                kernel_size -= 1
            if kernel_size > 1:
                soft_matte = TF.gaussian_blur(
                    matte,
                    kernel_size=[kernel_size, kernel_size],
                    sigma=kernel_size / 6.0,
                )
            else:
                soft_matte = matte
        else:
            soft_matte = matte

        # ── 5. 합성 ──
        z_out = z_pred * soft_matte + z_bg_noised * (1.0 - soft_matte)

        return z_out
