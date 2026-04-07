"""
End-to-end pipeline tests for HairS2INet (EHC v2).

All heavy components (VAE, transformer, ControlNet, text encoders) are mocked
so the test runs without GPU or pretrained weights.

Mock dimensions (scaled down from SD3.5 real):
  NUM_BLOCKS=6, INNER_DIM=64, H_LAT=W_LAT=32
  SEQ = (H_LAT//2) * (W_LAT//2) = 256  (matches MattePatchTokenizer AvgPool2d(2,2))
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# ── Mock constants ─────────────────────────────────────────────────────────────
NUM_BLOCKS = 6
INNER_DIM  = 64
H_LAT      = 32
W_LAT      = 32
SEQ        = (H_LAT // 2) * (W_LAT // 2)  # 256
B          = 2
H          = H_LAT * 8   # 256
W          = W_LAT * 8   # 256


# ── Mock classes ───────────────────────────────────────────────────────────────

class MockVAE(nn.Module):
    config = MagicMock()
    config.scaling_factor = 1.0

    def __init__(self, *a, **kw):
        super().__init__()

    def encode(self, x):
        B_ = x.shape[0]
        out = MagicMock()
        out.latent_dist.sample.return_value = torch.zeros(B_, 16, H_LAT, W_LAT)
        return out

    def decode(self, z):
        B_ = z.shape[0]
        out = MagicMock()
        out.sample = torch.zeros(B_, 3, H, W)
        return out

    def requires_grad_(self, flag):
        return self

    def eval(self):
        return self

    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()


class MockTransformerOutput:
    def __init__(self, sample):
        self.sample = sample


class MockTransformer(nn.Module):
    inner_dim = INNER_DIM
    config = MagicMock()
    config.num_train_timesteps = 1000

    def __init__(self, *a, **kw):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [nn.Linear(INNER_DIM, INNER_DIM) for _ in range(NUM_BLOCKS)]
        )

    def forward(self, hidden_states, **kwargs):
        B_ = hidden_states.shape[0]
        return MockTransformerOutput(torch.zeros(B_, 16, H_LAT, W_LAT))

    def requires_grad_(self, flag):
        for p in self.parameters():
            p.requires_grad_(flag)
        return self

    def enable_gradient_checkpointing(self):
        pass

    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()


class MockControlNetOutput:
    def __init__(self, residuals):
        self.controlnet_block_samples = residuals


class MockSD3ControlNet(nn.Module):
    def __init__(self, *a, **kw):
        super().__init__()
        self._dummy = nn.Linear(1, 1)  # so parameters() is non-empty

    def forward(self, hidden_states, controlnet_cond, **kwargs):
        B_ = hidden_states.shape[0]
        return MockControlNetOutput([
            torch.zeros(B_, SEQ, INNER_DIM) for _ in range(NUM_BLOCKS)
        ])

    @classmethod
    def from_pretrained(cls, *a, extra_conditioning_channels=0, **kw):
        return cls()


class MockTextEncoder(nn.Module):
    def __init__(self, *a, **kw):
        super().__init__()

    def forward(self, input_ids, output_hidden_states=False):
        B_ = input_ids.shape[0]
        out = MagicMock()
        out.hidden_states = [torch.zeros(B_, 77, 768)] * 13
        out.text_embeds   = torch.zeros(B_, 768)
        out.last_hidden_state = torch.zeros(B_, 256, 4096)
        return out

    def requires_grad_(self, flag):
        return self

    def eval(self):
        return self

    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()


class MockTokenizer:
    def __init__(self, *a, **kw):
        pass

    def __call__(self, text, **kw):
        out = MagicMock()
        out.input_ids = torch.zeros(1, 77, dtype=torch.long)
        return out

    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()


class MockScheduler:
    config = MagicMock()
    config.num_train_timesteps = 1000
    sigmas = torch.linspace(1, 0, 29)
    timesteps = torch.arange(28)

    def __init__(self, *a, **kw):
        pass

    def set_timesteps(self, *a, **kw):
        pass

    def step(self, noise_pred, t, z):
        out = MagicMock()
        out.prev_sample = z
        return out

    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def net():
    # create=True: diffusers 미설치 환경에서 모듈 네임스페이스에 속성이 없어도 patch 허용
    with patch("src.models.hair_s2i_net.AutoencoderKL", MockVAE, create=True), \
         patch("src.models.hair_s2i_net.SD3Transformer2DModel", MockTransformer, create=True), \
         patch("src.models.hair_s2i_net.SD3ControlNetModel", MockSD3ControlNet, create=True), \
         patch("src.models.hair_s2i_net.CLIPTextModelWithProjection", MockTextEncoder, create=True), \
         patch("src.models.hair_s2i_net.T5EncoderModel", MockTextEncoder, create=True), \
         patch("src.models.hair_s2i_net.CLIPTokenizer", MockTokenizer, create=True), \
         patch("src.models.hair_s2i_net.T5TokenizerFast", MockTokenizer, create=True), \
         patch("src.models.hair_s2i_net.FlowMatchEulerDiscreteScheduler", MockScheduler, create=True), \
         patch("src.models.hair_s2i_net.HAS_DIFFUSERS", True, create=True):
        from src.models.hair_s2i_net import HairS2INet
        model = HairS2INet("fake/path", blend_start_ratio=0.5)
        model.eval()
        return model


def _make_batch():
    return dict(
        background    = torch.randn(B, 3, H, W),
        sketch        = torch.rand(B, 1, H, W),
        matte         = torch.rand(B, 1, H, W),
        prompt_embeds = torch.zeros(B, 333, 4096),
        pooled_embeds = torch.zeros(B, 2048),
        timestep      = torch.ones(B, dtype=torch.long) * 500,
        target_latent = torch.randn(B, 16, H_LAT, W_LAT),
    )


# ── Forward shape tests ────────────────────────────────────────────────────────

class TestForwardShapes:
    def test_noise_pred_shape(self, net):
        batch = _make_batch()
        noise_pred, _ = net(**batch)
        assert noise_pred.shape == (B, 16, H_LAT, W_LAT)

    def test_returned_noise_shape(self, net):
        batch = _make_batch()
        _, noise = net(**batch)
        assert noise.shape == (B, 16, H_LAT, W_LAT)

    def test_external_noise_returned_unchanged(self, net):
        """Noise injected externally must be returned as-is (noise consistency)."""
        batch = _make_batch()
        ext_noise = torch.zeros(B, 16, H_LAT, W_LAT)
        _, noise_out = net(**batch, noise=ext_noise)
        assert torch.allclose(noise_out, ext_noise)

    def test_flow_target_same_shape(self, net):
        batch = _make_batch()
        _, noise = net(**batch)
        flow_target = noise - batch["target_latent"]
        assert flow_target.shape == (B, 16, H_LAT, W_LAT)


# ── ctrl_cond and MatteCNN tests ───────────────────────────────────────────────

class TestCtrlCondConstruction:
    def test_matte_cnn_zero_init_output(self, net):
        """MatteCNN zero-init: matte_feat must be ~0 before any training."""
        matte = torch.rand(B, 1, H, W)
        feat  = net.matte_cnn(matte)
        assert feat.shape == (B, 16, H_LAT, W_LAT)
        assert feat.abs().max().item() < 1e-6

    def test_matte_cnn_requires_grad(self, net):
        """matte_cnn must be trainable."""
        assert any(p.requires_grad for p in net.matte_cnn.parameters())

    def test_matte_tokenizer_no_params(self, net):
        assert sum(p.numel() for p in net.matte_tokenizer.parameters()) == 0


# ── Feature-Level Blending tests ───────────────────────────────────────────────

class TestFeatureLevelBlending:
    def test_early_blocks_untouched(self, net):
        """Early blocks (i < blend_start) must equal residuals exactly."""
        residuals    = [torch.ones(1, SEQ, INNER_DIM) * float(i + 1) for i in range(NUM_BLOCKS)]
        matte_tokens = torch.full((1, SEQ, 1), 0.3)
        blended      = net._blend(residuals, matte_tokens)

        blend_start = int(NUM_BLOCKS * net.blend_start_ratio)
        for i in range(blend_start):
            assert torch.allclose(blended[i], residuals[i]), \
                f"Block {i}: expected full residual, got gated"

    def test_late_blocks_matte_gated(self, net):
        """Late blocks (i >= blend_start) must equal residuals * matte_tokens."""
        residuals    = [torch.ones(1, SEQ, INNER_DIM) for _ in range(NUM_BLOCKS)]
        matte_tokens = torch.full((1, SEQ, 1), 0.5)
        blended      = net._blend(residuals, matte_tokens)

        blend_start = int(NUM_BLOCKS * net.blend_start_ratio)
        for i in range(blend_start, NUM_BLOCKS):
            expected = residuals[i] * matte_tokens
            assert torch.allclose(blended[i], expected), \
                f"Block {i}: expected matte-gated residual"

    def test_background_tokens_zeroed_by_gate(self, net):
        """matte=0 -> late-block residuals must be zeroed out."""
        residuals    = [torch.ones(1, SEQ, INNER_DIM) for _ in range(NUM_BLOCKS)]
        matte_tokens = torch.zeros(1, SEQ, 1)
        blended      = net._blend(residuals, matte_tokens)

        blend_start = int(NUM_BLOCKS * net.blend_start_ratio)
        for i in range(blend_start, NUM_BLOCKS):
            assert blended[i].abs().max().item() < 1e-6, \
                f"Block {i}: background tokens should be zeroed"

    def test_blend_preserves_count(self, net):
        residuals = [torch.zeros(1, SEQ, INNER_DIM) for _ in range(NUM_BLOCKS)]
        tokens    = torch.zeros(1, SEQ, 1)
        assert len(net._blend(residuals, tokens)) == NUM_BLOCKS

    def test_blend_output_shape(self, net):
        residuals    = [torch.randn(B, SEQ, INNER_DIM) for _ in range(NUM_BLOCKS)]
        matte_tokens = torch.rand(B, SEQ, 1)
        blended      = net._blend(residuals, matte_tokens)
        for b in blended:
            assert b.shape == (B, SEQ, INNER_DIM)


# ── Phase freeze tests ─────────────────────────────────────────────────────────

class TestPhase1Freeze:
    def test_transformer_frozen_after_freeze(self, net):
        net.freeze_transformer()
        trainable = any(p.requires_grad for p in net.transformer.parameters())
        assert not trainable, "Transformer must be frozen in Phase 1"

    def test_matte_cnn_trainable_after_freeze(self, net):
        net.freeze_transformer()
        trainable = any(p.requires_grad for p in net.matte_cnn.parameters())
        assert trainable, "matte_cnn must remain trainable in Phase 1"
