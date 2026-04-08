"""
HairS2INet — SD3.5 Hair Sketch-to-Image (EHC v2)

SketchHairSalon (SIGGRAPH Asia 2021) successor:
  MatteCNN        : matte soft alpha -> latent feature (trainable)
  SD3ControlNet   : official API, pos_embed handled internally
  Feature Blending: matte-gated token-level soft gate

ctrl_cond:
  sketch [B,3,H,W] -> RGB Colored Sketch -> Frozen VAE -> sketch_latent [B,16,64,64]
  matte  [B,1,H,W] -> MatteCNN               -> matte_feat   [B,16,64,64]
  matte            -> bilinear               -> matte_latent [B, 1,64,64]
  ctrl_cond = cat([sketch_latent + matte_feat, matte_latent]) -> [B,17,64,64]

Architecture Note:
- This model uses "Learned Null Embeddings" (nn.Parameter) instead of a text encoder.
- External text prompts are IGNORED to focus the model's capacity on sketch/mask control.

Feature-Level Blending (blend_start_ratio=0.5):
  residuals [B,1024,1152] x N blocks
  matte_tokens [B,1024,1]
  early blocks : blended[i] = residuals[i]              (global structure)
  late  blocks : blended[i] = residuals[i] * matte_tokens (soft matte gate)

diffusers 0.36.0 confirmed:
  SD3ControlNetModel(extra_conditioning_channels=1)
  -> controlnet_block_samples: List[Tensor[B,1024,1152]]
  PatchEmbed patch_size=2: [B,16,64,64] -> [B,1024,1152]
"""
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.matte_cnn import MatteCNN
from ..modules.matte_patch_tokenizer import MattePatchTokenizer
from ..modules.latent_compositor import TimestepAwareLatentCompositor
from ..utils.preprocess import (
    preprocess_image, preprocess_sketch, preprocess_matte,
    postprocess_image,
)

try:
    from diffusers import (
        AutoencoderKL,
        SD3Transformer2DModel,
        SD3ControlNetModel,
        FlowMatchEulerDiscreteScheduler,
    )
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False


class HairS2INet(nn.Module):
    """
    SD3.5 Hair Sketch-to-Image model.

    Training : forward() -> (noise_pred, noise), trainer computes flow_target
    Inference: inference() -> PIL Image
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        blend_start_ratio: float = 0.5,
    ):
        super().__init__()
        if not HAS_DIFFUSERS:
            raise ImportError("diffusers and transformers are required.")

        self.blend_start_ratio = blend_start_ratio

        # SD3.5 components (bf16으로 로드 → VRAM 절반)
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        # Freeze VAE (transformer remains unfrozen unless freeze_transformer() called)
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.vae_scale_factor = self.vae.config.scaling_factor
        self.vae_shift_factor = getattr(self.vae.config, "shift_factor", 0.0609)

        # Learned Null Embedding — 텍스트 인코더 대체용 고정 신호
        # SD3.5 Medium 규격: prompt_embeds=[1, 333, 4096], pooled=[1, 2048]
        self.null_encoder_hidden_states = nn.Parameter(
            torch.zeros(1, 333, 4096, dtype=torch.bfloat16)
        )
        self.null_pooled_projections = nn.Parameter(
            torch.zeros(1, 2048, dtype=torch.bfloat16)
        )

        # Trainable modules
        # MatteCNN: zero-init -> matte_feat~=0 at init -> sane start
        self.matte_cnn = MatteCNN(out_channels=16).to(torch.bfloat16)

        # MM-DiT Base Transformer (Frozen or partially trainable)
        self.transformer = SD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

        # SD3ControlNetModel initialized from transformer weights (12 layers for B8 safety)
        # We manually set extra_conditioning_channels=1 to handle 16ch sketch + 1ch matte
        self.sd3_controlnet = SD3ControlNetModel.from_transformer(
            self.transformer,
            num_layers=12,
        )
        # Ensure correct bfloat16 dtype
        self.sd3_controlnet.to(torch.bfloat16)
        # Manually patching conditioning channels if not set (legacy/version compatibility)
        if self.sd3_controlnet.config.extra_conditioning_channels != 1:
            self.sd3_controlnet.config.extra_conditioning_channels = 1
            # Re-initialize pos_embed_input to accept 17 channels (16 + 1)
            inner_dim = self.transformer.config.joint_attention_dim
            self.sd3_controlnet.pos_embed_input = nn.Linear(16 + 1, inner_dim).to(
                device=self.transformer.device, dtype=torch.bfloat16
            )
            nn.init.zeros_(self.sd3_controlnet.pos_embed_input.weight)
            nn.init.zeros_(self.sd3_controlnet.pos_embed_input.bias)

        # Scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
        )

        # No-parameter modules
        self.matte_tokenizer = MattePatchTokenizer()
        self.compositor = TimestepAwareLatentCompositor()

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        background:    torch.Tensor,
        sketch:        torch.Tensor,
        matte:         torch.Tensor,
        timestep:      torch.Tensor,
        target_latent: torch.Tensor,
        noise:         Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            background:    [B, 3, H, W]
            sketch:        [B, 1, H, W]
            matte:         [B, 1, H, W]
            timestep:      [B]
            target_latent: [B, 16, 64, 64]
            noise:         optional noise for z_noisy
        """
        B = background.shape[0]
        # Learned null embeddings expanded to batch size
        prompt_embeds = self.null_encoder_hidden_states.expand(B, -1, -1)
        pooled_embeds = self.null_pooled_projections.expand(B, -1)
        # 1. Build ctrl_cond
        with torch.no_grad():
            # Encoding: (raw - shift) * scale
            sketch_latent = self.vae.encode(sketch).latent_dist.sample()
            sketch_latent = (sketch_latent - self.vae_shift_factor) * self.vae_scale_factor

        matte_feat   = self.matte_cnn(matte)                    # [B,16,H/8,W/8]  trainable
        matte_latent = F.interpolate(
            matte, size=sketch_latent.shape[-2:],
            mode='bilinear', align_corners=False,
        )                                                       # [B,1,H/8,W/8]
        ctrl_cond = torch.cat(
            [sketch_latent + matte_feat, matte_latent], dim=1
        )                                                       # [B,17,H/8,W/8]

        # 2. Flow Matching noising
        if noise is None:
            noise = torch.randn_like(target_latent)
        sigma   = self._timestep_to_sigma(timestep).view(-1, 1, 1, 1)
        z_noisy = (1.0 - sigma) * target_latent + sigma * noise

        # 3. ControlNet residuals
        controlnet_out = self.sd3_controlnet(
            hidden_states=z_noisy,
            controlnet_cond=ctrl_cond,
            conditioning_scale=1.0,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            timestep=timestep,
        )
        residuals    = controlnet_out.controlnet_block_samples  # List[Tensor[B,seq,1152]]

        # 4. Feature-Level Matte-Gated Blending
        matte_tokens = self.matte_tokenizer(matte_latent)       # [B,seq,1]
        blended      = self._blend(residuals, matte_tokens)

        # 5. MM-DiT forward
        noise_pred = self.transformer(
            hidden_states=z_noisy,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            timestep=timestep,
            block_controlnet_hidden_states=blended,
        ).sample

        return noise_pred, noise

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference(
        self,
        background:     "PIL.Image.Image",
        sketch:         "PIL.Image.Image",
        matte:          "PIL.Image.Image",
        prompt:         Optional[str] = None,
        num_steps:      int   = 28,
        guidance_scale: float = 7.0,
        size:           Tuple[int, int] = (512, 512),
        seed:           Optional[int] = None,
    ) -> "PIL.Image.Image":
        """
        Single-image inference. ctrl_cond computed once outside the loop.
        CFG: uncond uses zero residuals to amplify sketch control.
        """
        device = next(self.parameters()).device

        if seed is not None:
            torch.manual_seed(seed)

        bg_tensor = preprocess_image(background, size, device).to(torch.bfloat16)
        sk_tensor = preprocess_sketch(sketch, size, device).to(torch.bfloat16)
        mt_tensor = preprocess_matte(matte, size, device).to(torch.bfloat16)

        # prompt encoding 제거 -> null_parameters 사용
        prompt_embeds = self.null_encoder_hidden_states
        pooled_embeds = self.null_pooled_projections

        # Encoding: (raw - shift) * scale
        z_bg = self.vae.encode(bg_tensor).latent_dist.sample().to(torch.bfloat16)
        z_bg = (z_bg - self.vae_shift_factor) * self.vae_scale_factor

        # Compute ctrl_cond once (sketch/matte fixed across steps)
        sketch_latent = self.vae.encode(sk_tensor).latent_dist.sample().to(torch.bfloat16)
        sketch_latent = (sketch_latent - self.vae_shift_factor) * self.vae_scale_factor
        matte_feat   = self.matte_cnn(mt_tensor)
        mt_latent    = F.interpolate(
            mt_tensor, size=z_bg.shape[-2:], mode='bilinear', align_corners=False
        )
        ctrl_cond    = torch.cat([sketch_latent + matte_feat, mt_latent], dim=1)
        matte_tokens = self.matte_tokenizer(mt_latent)

        z = torch.randn_like(z_bg)
        self.scheduler.set_timesteps(num_steps, device=device)

        for step_idx, t in enumerate(self.scheduler.timesteps):
            t_batch = t.unsqueeze(0).to(device)

            residuals_cond = self.sd3_controlnet(
                hidden_states=z,
                controlnet_cond=ctrl_cond,
                conditioning_scale=1.0,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                timestep=t_batch,
            ).controlnet_block_samples

            blended_cond = self._blend(residuals_cond, matte_tokens)

            noise_pred_cond = self.transformer(
                hidden_states=z,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                timestep=t_batch,
                block_controlnet_hidden_states=blended_cond,
            ).sample

            noise_pred_uncond = self.transformer(
                hidden_states=z,
                encoder_hidden_states=prompt_embeds,  # uncond도 null 사용
                pooled_projections=pooled_embeds,
                timestep=t_batch,
                block_controlnet_hidden_states=[
                    torch.zeros_like(r) for r in residuals_cond
                ],
            ).sample

            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            z = self.scheduler.step(noise_pred, t, z).prev_sample

            sigma = self.scheduler.sigmas[step_idx].to(device)
            z = self.compositor(z, z_bg, mt_latent, sigma.unsqueeze(0).expand(1))

        # Decoding: (latent / scale) + shift
        z_raw = (z / self.vae_scale_factor) + self.vae_shift_factor
        image = self.vae.decode(z_raw).sample
        return postprocess_image(image)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _blend(
        self,
        residuals:    List[torch.Tensor],
        matte_tokens: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Feature-Level Matte-Gated Blending.
        early blocks (i < blend_start): full residual  -> global hair structure
        late  blocks (i >= blend_start): r * matte_tokens -> soft matte gate
        broadcast: [B,seq,1152] * [B,seq,1] -> [B,seq,1152]
        """
        blend_start = int(len(residuals) * self.blend_start_ratio)
        return [
            r if i < blend_start else r * matte_tokens
            for i, r in enumerate(residuals)
        ]

    # encode_prompt 제거

    def _timestep_to_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        num_train = self.scheduler.config.num_train_timesteps
        return timestep.float() / num_train

    def freeze_transformer(self):
        self.transformer.requires_grad_(False)

    def unfreeze_transformer_last_n_blocks(self, n: int):
        self.transformer.requires_grad_(False)
        for block in list(self.transformer.transformer_blocks)[-n:]:
            block.requires_grad_(True)

    def unfreeze_transformer_all(self):
        self.transformer.requires_grad_(True)
