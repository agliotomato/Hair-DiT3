"""
HairS2INet — SD3.5 Hair Sketch-to-Image (EHC v2)

SketchHairSalon (SIGGRAPH Asia 2021) successor:
  MatteCNN        : matte soft alpha -> latent feature (trainable)
  SD3ControlNet   : official API, pos_embed handled internally
  Feature Blending: matte-gated token-level soft gate

ctrl_cond:
  sketch [B,1,H,W] -> repeat(3) -> Frozen VAE -> sketch_latent [B,16,64,64]
  matte  [B,1,H,W] -> MatteCNN               -> matte_feat   [B,16,64,64]
  matte            -> bilinear               -> matte_latent [B, 1,64,64]
  ctrl_cond = cat([sketch_latent + matte_feat, matte_latent]) -> [B,17,64,64]

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
    from transformers import (
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        T5EncoderModel,
        T5TokenizerFast,
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
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer_2"
        )
        self.tokenizer_3 = T5TokenizerFast.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer_3"
        )
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16,
        )
        self.text_encoder_3 = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder_3",
            torch_dtype=torch.bfloat16,
        )
        self.transformer = SD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )

        # Freeze VAE and text encoders
        for model in [self.vae, self.text_encoder, self.text_encoder_2, self.text_encoder_3]:
            model.requires_grad_(False)
            model.eval()

        self.vae_scale_factor = self.vae.config.scaling_factor

        # Trainable modules
        # MatteCNN: zero-init -> matte_feat~=0 at init -> sane start
        self.matte_cnn = MatteCNN(out_channels=16)

        # SD3ControlNetModel initialized from transformer weights
        # extra_conditioning_channels=1 -> pos_embed_input accepts 17ch (zero-init internally)
        self.sd3_controlnet = SD3ControlNetModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
            extra_conditioning_channels=1,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
            torch_dtype=torch.bfloat16,
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
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        timestep:      torch.Tensor,
        target_latent: torch.Tensor,
        noise:         Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            background:    [B, 3, H, W]  in [-1, 1]
            sketch:        [B, 1, H, W]  in [0, 1]
            matte:         [B, 1, H, W]  in [0, 1]
            prompt_embeds: [B, seq, 4096]
            pooled_embeds: [B, 2048]
            timestep:      [B] int
            target_latent: [B, 16, H/8, W/8]
            noise:         [B, 16, H/8, W/8] optional external noise for consistency
        Returns:
            noise_pred: [B, 16, H/8, W/8]
            noise:      [B, 16, H/8, W/8]  same noise used for z_noisy
        """
        # 1. Build ctrl_cond
        with torch.no_grad():
            sketch_latent = self.vae.encode(sketch).latent_dist.sample() * self.vae_scale_factor      # [B,16,H/8,W/8]

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
        prompt:         str,
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

        bg_tensor = preprocess_image(background, size, device)
        sk_tensor = preprocess_sketch(sketch, size, device)
        mt_tensor = preprocess_matte(matte, size, device)

        prompt_embeds, pooled_embeds = self.encode_prompt(prompt, device)
        uncond_embeds, uncond_pooled = self.encode_prompt("", device)

        z_bg = self.vae.encode(bg_tensor).latent_dist.sample() * self.vae_scale_factor

        # Compute ctrl_cond once (sketch/matte fixed across steps)
        sketch_latent = self.vae.encode(sk_tensor).latent_dist.sample() * self.vae_scale_factor
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
                encoder_hidden_states=uncond_embeds,
                pooled_projections=uncond_pooled,
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

        image = self.vae.decode(z / self.vae_scale_factor).sample
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

    def encode_prompt(
        self,
        prompt: str,
        device: torch.device,
        max_sequence_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            prompt_embeds: [1, 333, 4096]
            pooled_embeds: [1, 2048]
        """
        tokens_1 = self.tokenizer(
            prompt, return_tensors="pt",
            padding="max_length", truncation=True, max_length=77
        ).input_ids.to(device)
        enc_out_1     = self.text_encoder(tokens_1, output_hidden_states=True)
        clip_l_emb    = enc_out_1.hidden_states[-2]
        pooled_clip_l = enc_out_1.text_embeds

        tokens_2 = self.tokenizer_2(
            prompt, return_tensors="pt",
            padding="max_length", truncation=True, max_length=77
        ).input_ids.to(device)
        enc_out_2     = self.text_encoder_2(tokens_2, output_hidden_states=True)
        clip_g_emb    = enc_out_2.hidden_states[-2]
        pooled_clip_g = enc_out_2.text_embeds

        tokens_3 = self.tokenizer_3(
            prompt, return_tensors="pt",
            padding="max_length", truncation=True, max_length=max_sequence_length
        ).input_ids.to(device)
        t5_emb = self.text_encoder_3(tokens_3).last_hidden_state

        clip_emb      = torch.cat([clip_l_emb, clip_g_emb], dim=-1)
        clip_emb      = F.pad(clip_emb, (0, t5_emb.shape[-1] - clip_emb.shape[-1]))
        prompt_embeds = torch.cat([clip_emb, t5_emb], dim=1)
        pooled_embeds = torch.cat([pooled_clip_l, pooled_clip_g], dim=-1)

        return prompt_embeds, pooled_embeds

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
