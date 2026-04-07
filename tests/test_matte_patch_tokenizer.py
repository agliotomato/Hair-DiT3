"""Tests for MattePatchTokenizer"""
import pytest
import torch
from src.modules.matte_patch_tokenizer import MattePatchTokenizer


class TestMattePatchTokenizer:
    def test_output_shape_sd35(self):
        """SD3.5 standard: latent 64x64 -> 1024 tokens."""
        tok = MattePatchTokenizer()
        matte_latent = torch.rand(2, 1, 64, 64)
        tokens = tok(matte_latent)
        # AvgPool2d(2,2): 64->32, seq = 32*32 = 1024
        assert tokens.shape == (2, 1024, 1)

    def test_output_shape_generic(self):
        tok = MattePatchTokenizer()
        matte_latent = torch.rand(3, 1, 32, 32)
        tokens = tok(matte_latent)
        assert tokens.shape == (3, 256, 1)  # 16*16=256

    def test_no_parameters(self):
        """MattePatchTokenizer must have zero learnable parameters."""
        tok = MattePatchTokenizer()
        total_params = sum(p.numel() for p in tok.parameters())
        assert total_params == 0

    def test_value_range_preserved(self):
        """AvgPool of [0,1] values must stay in [0,1]."""
        tok = MattePatchTokenizer()
        matte_latent = torch.rand(1, 1, 64, 64)  # [0,1]
        tokens = tok(matte_latent)
        assert tokens.min().item() >= 0.0 - 1e-6
        assert tokens.max().item() <= 1.0 + 1e-6

    def test_broadcast_with_residuals(self):
        """blended = residuals * matte_tokens must broadcast correctly."""
        tok = MattePatchTokenizer()
        matte_latent = torch.rand(2, 1, 64, 64)
        tokens = tok(matte_latent)           # [2, 1024, 1]
        residual = torch.randn(2, 1024, 1152)
        blended = residual * tokens          # broadcast -> [2, 1024, 1152]
        assert blended.shape == (2, 1024, 1152)

    def test_hair_tokens_close_to_one(self):
        """Tokens at hair positions (matte=1) should be ~1."""
        tok = MattePatchTokenizer()
        matte_latent = torch.ones(1, 1, 64, 64)
        tokens = tok(matte_latent)
        assert tokens.min().item() > 0.99

    def test_background_tokens_close_to_zero(self):
        """Tokens at background positions (matte=0) should be ~0."""
        tok = MattePatchTokenizer()
        matte_latent = torch.zeros(1, 1, 64, 64)
        tokens = tok(matte_latent)
        assert tokens.max().item() < 0.01

    def test_blending_gate_effect(self):
        """With matte=0, blended residual must be zeroed out."""
        tok = MattePatchTokenizer()
        matte_latent = torch.zeros(1, 1, 64, 64)
        tokens = tok(matte_latent)                   # all ~0
        residual = torch.ones(1, 1024, 1152)
        blended = residual * tokens
        assert blended.abs().max().item() < 1e-6
