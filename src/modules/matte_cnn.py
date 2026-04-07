"""
MatteCNN
matte (1ch soft alpha) → matte_feat (16ch latent-space feature)

SketchHairSalon Stage 1 출력(soft alpha matte)을 VAE latent 공간과 같은
feature 차원으로 변환. sketch_latent(Frozen VAE)와 element-wise ADD로 합산되어
ctrl_cond의 16ch 앞 부분을 구성한다.

Zero-init 효과:
    학습 초기 matte_feat ≈ 0
    → sketch_latent + matte_feat ≈ sketch_latent
    → 사전학습 분포에서 ControlNet이 안정적으로 시작
"""
import torch
import torch.nn as nn


class MatteCNN(nn.Module):
    """
    matte [B, 1, H, W] → matte_feat [B, out_channels, H/8, W/8]

    구조:
        Conv(1→32, stride=2) → SiLU  : H/2
        Conv(32→64, stride=2) → SiLU : H/4
        Conv(64→C, stride=2)          : H/8  (마지막 Conv: Zero-init, SiLU 없음)

    Zero-init (마지막 Conv weight/bias = 0):
        SiLU 없이 Zero-init → 초기 출력이 정확히 0
        (SketchConditionEncoder는 Zero-init 후 SiLU를 붙였으나 SiLU(0)=0으로 동일)
    """

    def __init__(self, out_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # H/2
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # H/4
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),  # H/8
            # SiLU 없음: Zero-init 직후 출력 = 0 을 명시적으로 보장
        )
        # 마지막 Conv2d (index 4) Zero-init
        nn.init.zeros_(self.encoder[4].weight)
        nn.init.zeros_(self.encoder[4].bias)

    def forward(self, matte: torch.Tensor) -> torch.Tensor:
        """
        Args:
            matte: [B, 1, H, W] float32, 값 범위 [0, 1]
        Returns:
            matte_feat: [B, out_channels, H/8, W/8]
        """
        return self.encoder(matte)
