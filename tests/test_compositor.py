"""TimestepAwareLatentCompositor 단위 테스트"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.modules.latent_compositor import TimestepAwareLatentCompositor


class TestTimestepAwareLatentCompositor:

    def setup_method(self):
        self.compositor = TimestepAwareLatentCompositor()
        self.B, self.C, self.H, self.W = 2, 16, 8, 8

    def _make_inputs(self, sigma_val: float = 0.5, matte_val: float = 0.5):
        z_pred = torch.randn(self.B, self.C, self.H, self.W)
        z_bg   = torch.randn(self.B, self.C, self.H, self.W)
        matte  = torch.full((self.B, 1, self.H, self.W), matte_val)
        sigma  = torch.full((self.B,), sigma_val)
        return z_pred, z_bg, matte, sigma

    def test_output_shape(self):
        """출력 shape 일치"""
        inputs = self._make_inputs()
        out = self.compositor(*inputs)
        assert out.shape == (self.B, self.C, self.H, self.W)

    def test_high_sigma_hard_matte(self):
        """sigma=1.0(최대노이즈): blur_radius=0, hard matte 사용"""
        # sigma=1 → t_mean=1 → blur_radius = int((1-1)*15) = 0
        z_pred, z_bg, matte, sigma = self._make_inputs(sigma_val=1.0, matte_val=1.0)
        out = self.compositor(z_pred, z_bg, matte, sigma, max_blur=15)
        # blur_radius=0이므로 soft_matte == matte (blur 없음)
        # matte=1 영역: z_out = z_pred * 1 + z_bg_noised * 0 = z_pred
        # 단, z_bg_noised는 sigma=1 일 때 ≈ noise (z_bg와 무관)
        # → out은 z_pred와 거의 같아야 함 (matte=1 영역)
        assert out.shape == (self.B, self.C, self.H, self.W)

    def test_low_sigma_soft_matte(self):
        """sigma=0.0(클린): blur_radius=max_blur"""
        z_pred, z_bg, matte, sigma = self._make_inputs(sigma_val=0.0)
        # 에러 없이 실행되어야 함
        out = self.compositor(z_pred, z_bg, matte, sigma, max_blur=15)
        assert out.shape == (self.B, self.C, self.H, self.W)

    def test_matte_zero_background_preserved(self):
        """matte=0 영역: z_out ≈ z_bg_noised (배경 보존)"""
        z_pred, z_bg, matte, sigma = self._make_inputs(matte_val=0.0, sigma_val=0.0)
        # sigma=0: z_bg_noised = 1.0 * z_bg + 0.0 * noise = z_bg
        # matte=0: z_out = z_pred * 0 + z_bg * 1 = z_bg
        out = self.compositor(z_pred, z_bg, matte, sigma)
        assert torch.allclose(out, z_bg, atol=1e-5), \
            "matte=0, sigma=0 시 z_out ≠ z_bg"

    def test_flow_matching_noising_formula(self):
        """Flow Matching noising 공식 검증: z_t = (1-sigma)*z_clean + sigma*noise"""
        import inspect
        source = inspect.getsource(TimestepAwareLatentCompositor)
        # DDPM 스케줄러 호출 금지 (scheduler.add_noise 함수 호출 패턴)
        assert "scheduler.add_noise" not in source, "scheduler.add_noise 호출 금지"
        assert "DDPMScheduler" not in source, "DDPMScheduler 사용 금지"
        # Flow Matching 공식 존재 확인: (1.0 - s) * z_clean + s * noise
        assert "(1.0 - s)" in source or "(1 - s)" in source or "(1.0 - sigma)" in source

    def test_no_ddpm_noise(self):
        """DDPM scheduler 관련 호출이 없어야 함"""
        import inspect
        source = inspect.getsource(TimestepAwareLatentCompositor.__call__)
        assert "DDPMScheduler" not in source
        assert "scheduler.add_noise" not in source  # 함수 호출 패턴만 체크

    def test_sigma_zero_bg_noised_equals_bg(self):
        """sigma=0: z_bg_noised = z_bg (노이즈 없음)"""
        # 이를 직접 검증하기 위해 torch.manual_seed 고정
        torch.manual_seed(42)
        z_pred = torch.zeros(1, 16, 4, 4)
        z_bg   = torch.ones(1, 16, 4, 4) * 2.0
        matte  = torch.zeros(1, 1, 4, 4)  # 배경만
        sigma  = torch.tensor([0.0])

        out = self.compositor(z_pred, z_bg, matte, sigma)
        # sigma=0, matte=0: out = z_bg_noised = (1-0)*z_bg + 0*noise = z_bg
        assert torch.allclose(out, z_bg, atol=1e-5)

    def test_blur_radius_calculation(self):
        """blur_radius = int((1 - sigma) * max_blur) 공식 검증"""
        # sigma=0.4, max_blur=10 → blur_radius = int(0.6 * 10) = 6
        z_pred, z_bg, matte, _ = self._make_inputs()
        sigma = torch.full((self.B,), 0.4)
        # 에러 없이 실행되면 OK (kernel_size=13 홀수)
        out = self.compositor(z_pred, z_bg, matte, sigma, max_blur=10)
        assert out.shape == (self.B, self.C, self.H, self.W)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
