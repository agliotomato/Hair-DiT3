"""
MattePatchTokenizer
matte_latent → token sequence (SD3 PatchEmbed 해상도에 맞춤)

SD3.5 MM-DiT는 pos_embed(PatchEmbed, patch_size=2)를 통해
latent [B, 16, 64, 64] → sequence [B, 1024, 1152] 로 변환한다.
(32×32 = 1024 tokens)

MattePatchTokenizer는 동일한 patch_size=2 기준으로 matte를 토큰화:
    matte_latent [B, 1, 64, 64]
    → AvgPool2d(kernel=2, stride=2) → [B, 1, 32, 32]
    → flatten + transpose          → matte_tokens [B, 1024, 1]

Feature-Level Blending 공식:
    전반 블록 (i < blend_start): blended[i] = residuals[i]            # full residual
    후반 블록 (i >= blend_start): blended[i] = residuals[i] * tokens  # soft matte gate
    broadcast: [B, 1024, 1152] * [B, 1024, 1] → [B, 1024, 1152]

파라미터 없음: AvgPool2d는 학습 파라미터를 갖지 않는다.
"""
import torch
import torch.nn as nn


class MattePatchTokenizer(nn.Module):
    """
    matte_latent [B, 1, H_lat, W_lat] → matte_tokens [B, seq, 1]

    seq = (H_lat / 2) * (W_lat / 2)
    SD3.5 기준 (512×512 입력): H_lat=W_lat=64 → seq = 32×32 = 1024

    파라미터 없음 (purely functional).
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, matte_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            matte_latent: [B, 1, H_lat, W_lat] float32 in [0, 1]
        Returns:
            matte_tokens: [B, seq, 1]
                seq = (H_lat/2) * (W_lat/2)
                헤어 token ≈ 1.0, 배경 token ≈ 0.0
        """
        pooled = self.pool(matte_latent)           # [B, 1, H/2, W/2]
        tokens = pooled.flatten(2)                 # [B, 1, seq]
        return tokens.transpose(1, 2)             # [B, seq, 1]
