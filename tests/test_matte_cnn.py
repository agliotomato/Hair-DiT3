"""Tests for MatteCNN"""
import pytest
import torch
from src.modules.matte_cnn import MatteCNN


class TestMatteCNN:
    def test_output_shape_512(self):
        cnn = MatteCNN(out_channels=16)
        x = torch.zeros(2, 1, 512, 512)
        out = cnn(x)
        assert out.shape == (2, 16, 64, 64)

    def test_output_shape_custom_channels(self):
        cnn = MatteCNN(out_channels=8)
        x = torch.zeros(1, 1, 256, 256)
        out = cnn(x)
        assert out.shape == (1, 8, 32, 32)

    def test_zero_init_output(self):
        """Zero-init last conv -> output must be exactly 0 for any input."""
        cnn = MatteCNN(out_channels=16)
        x = torch.randn(1, 1, 64, 64)
        out = cnn(x)
        assert out.abs().max().item() < 1e-6

    def test_last_conv_weight_zero(self):
        cnn = MatteCNN(out_channels=16)
        assert cnn.encoder[4].weight.abs().max().item() < 1e-8

    def test_last_conv_bias_zero(self):
        cnn = MatteCNN(out_channels=16)
        assert cnn.encoder[4].bias.abs().max().item() < 1e-8

    def test_gradient_flows_through_early_convs(self):
        cnn = MatteCNN(out_channels=16)
        # Manually set last conv weights to non-zero to allow gradient flow
        torch.nn.init.kaiming_uniform_(cnn.encoder[4].weight)
        x = torch.randn(1, 1, 64, 64)
        cnn(x).sum().backward()
        assert cnn.encoder[0].weight.grad is not None
        assert cnn.encoder[0].weight.grad.abs().max().item() > 0

    def test_no_global_pooling(self):
        """Spatial dimensions must be preserved proportionally (no global pool)."""
        cnn = MatteCNN(out_channels=16)
        x = torch.randn(1, 1, 512, 512)
        out = cnn(x)
        # stride=2 x3 -> H/8, W/8
        assert out.shape[-2] == 512 // 8
        assert out.shape[-1] == 512 // 8

    def test_matte_value_range_input(self):
        """Model should not crash on [0,1] range input."""
        cnn = MatteCNN(out_channels=16)
        x = torch.rand(2, 1, 128, 128)  # values in [0,1]
        out = cnn(x)
        assert out.shape == (2, 16, 16, 16)
        assert torch.isfinite(out).all()
