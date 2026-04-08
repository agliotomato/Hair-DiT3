"""
학습 루프 — EHC v2

학습 대상: matte_cnn + sd3_controlnet (transformer 전체 동결)
ControlNet 패러다임: 베이스 모델(SD3.5 transformer)은 건드리지 않음.

HAIR-DIT 이식 기법:
  - Logit-Normal sigma 샘플링: 중간 노이즈 레벨 집중 학습
  - Linear LR Warmup: 초반 warmup_steps 동안 선형 증가
  - EMA: 학습 파라미터 이동 평균 → 추론 시 안정적인 가중치
"""
import logging
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import HairRegionDataset, HairS2IDataset
from ..models.hair_s2i_net import HairS2INet
from ..losses.hair_s2i_loss import HairS2ILoss

logger = logging.getLogger(__name__)


class Trainer:
    """EHC 모델 학습 루프. cfg는 YAML에서 로드한 dict."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _sample_sigmas(self, bsz: int, scheduler) -> torch.Tensor:
        """
        Logit-Normal 분포로 타임스텝 샘플링.
        Flow Matching에서 중간 sigma(~0.5)가 학습 난이도가 가장 높음.
        """
        t = self.cfg["training"]
        n_train = scheduler.config.num_train_timesteps
        u = torch.sigmoid(
            torch.normal(
                mean=t["logit_mean"],
                std=t["logit_std"],
                size=(bsz,),
                device=self.device,
            )
        )
        return (u * n_train).long().clamp(1, n_train - 1)

    def train(self):
        cfg     = self.cfg
        t       = cfg["training"]
        ckpt    = cfg["checkpointing"]
        model_c = cfg["model"]
        data_c  = cfg["data"]
        lw      = t["loss_weights"]

        # ── 모델 초기화 ──
        logger.info(f"모델 로드: {model_c['model_id']}")
        model = HairS2INet(
            model_c["model_id"],
            blend_start_ratio=model_c["blend_start_ratio"],
        ).to(self.device)

        model.freeze_transformer()

        if t.get("gradient_checkpointing", True):
            model.transformer.enable_gradient_checkpointing()

        # 체크포인트 로드
        resume_from = t.get("resume_from")
        if resume_from:
            logger.info(f"체크포인트 로드: {resume_from}")
            state = torch.load(resume_from, map_location=self.device)
            if "matte_cnn" in state:
                model.matte_cnn.load_state_dict(state["matte_cnn"])
                model.sd3_controlnet.load_state_dict(state["sd3_controlnet"])
            elif "sketch_encoder" in state:
                logger.warning("구 체크포인트 형식 (sketch_encoder) — 무시하고 새로 학습합니다.")
            else:
                model.load_state_dict(state, strict=False)

        # ── 데이터셋 ──
        dataset_split = t["dataset"]
        if dataset_split in HairRegionDataset.VALID_SPLITS:
            train_dataset = HairRegionDataset(
                split=dataset_split,
                dataset_root=data_c["root"],
                image_size=data_c["image_size"],
                augment=True,
                target_split=t.get("target_split"),
            )
        else:
            train_dataset = HairS2IDataset(
                data_c["root"], data_c["image_size"], augment=True
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=t["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        epochs      = t["epochs"]
        total_steps = epochs * len(train_loader)

        # ── 손실 함수 ──
        criterion = HairS2ILoss(
            phase=t.get("phase", 1),
            lambda_bg=lw.get("bg", 3.0),
            lambda_lpips=lw.get("lpips", 0.1),
            lambda_edge=lw.get("edge", 0.05),
        ).to(self.device)

        # ── Optimizer & LR Scheduler ──
        optimizer = AdamW(
            [
                {"params": model.matte_cnn.parameters()},
                {"params": model.sd3_controlnet.parameters()},
                {"params": [model.null_encoder_hidden_states, model.null_pooled_projections]},
            ],
            lr=t["learning_rate"],
            weight_decay=1e-2,
        )
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

        # ── EMA ──
        ema_model = None
        ema_decay = t.get("ema_decay")
        if ema_decay:
            try:
                from torch_ema import ExponentialMovingAverage
                ema_model = ExponentialMovingAverage(
                    [p for p in model.parameters() if p.requires_grad],
                    decay=ema_decay,
                )
                logger.info(f"EMA 활성화: decay={ema_decay}")
            except ImportError:
                logger.warning("torch_ema 미설치 → EMA 생략.")

        # ── Mixed Precision ──
        scaler = None
        autocast_dtype = torch.float32
        mp = t.get("mixed_precision", "no")
        if mp == "bf16" and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        elif mp == "fp16":
            scaler = torch.cuda.amp.GradScaler()
            autocast_dtype = torch.float16

        grad_accum  = t.get("gradient_accumulation_steps", 1)
        warmup_steps = t.get("warmup_steps", 0)
        grad_clip   = t.get("gradient_clip", 1.0)
        log_every   = ckpt.get("log_every", 50)
        save_every  = ckpt.get("save_every", 1000)
        output_dir  = ckpt["output_dir"]

        # ── 학습 루프 ──
        global_step    = 0
        grad_accum_cnt = 0
        running_loss: Dict[str, float] = {}

        model.train()
        logger.info(f"학습 시작: {epochs} epochs / {total_steps} steps")

        for epoch in range(epochs):
            for batch in train_loader:
                background = batch["background"].to(self.device, dtype=torch.float32)
                sketch     = batch["sketch"].to(self.device, dtype=torch.float32)
                matte      = batch["matte"].to(self.device, dtype=torch.float32)
                target_img = batch["target"].to(self.device, dtype=torch.float32)

                B        = background.shape[0]
                timestep = self._sample_sigmas(B, model.scheduler)

                with torch.no_grad():
                    vae_dtype = next(model.vae.parameters()).dtype
                    z_target = model.vae.encode(target_img.to(vae_dtype)).latent_dist.sample()
                    z_target = (z_target.float() - model.vae_shift_factor) * model.vae_scale_factor

                noise = torch.randn_like(z_target)

                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    noise_pred, noise = model(
                        background=background,
                        sketch=sketch,
                        matte=matte,
                        timestep=timestep,
                        target_latent=z_target,
                        noise=noise,
                    )

                    flow_target = noise - z_target

                    with torch.no_grad():
                        z_bg = model.vae.encode(background.to(vae_dtype)).latent_dist.sample()
                        z_bg = (z_bg.float() - model.vae_shift_factor) * model.vae_scale_factor

                    from ..utils.preprocess import matte_to_latent
                    matte_latent = matte_to_latent(matte, z_bg.shape[-2:])

                    pred_z0 = noise - noise_pred

                    pred_image = None
                    if lw.get("lpips", 0) > 0 or lw.get("edge", 0) > 0:
                        pred_image = model.vae.decode(
                            ((pred_z0 / model.vae_scale_factor) + model.vae_shift_factor).to(vae_dtype)
                        ).sample.float()

                    total_loss, loss_dict = criterion(
                        noise_pred=noise_pred,
                        noise_target=flow_target,
                        pred_latent=pred_z0,
                        z_bg=z_bg,
                        matte_latent=matte_latent,
                        pred_image=pred_image,
                        target_image=target_img,
                        sketch=sketch,
                        matte_image=matte,
                        global_step=global_step,
                        total_steps=total_steps,
                    )
                    total_loss = total_loss / grad_accum

                if scaler is not None:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

                grad_accum_cnt += 1

                if grad_accum_cnt >= grad_accum:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        grad_clip,
                    )
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    # LR Warmup → Cosine decay
                    if global_step < warmup_steps:
                        lr_scale = (global_step + 1) / max(warmup_steps, 1)
                        for pg in optimizer.param_groups:
                            if "initial_lr" not in pg:
                                pg["initial_lr"] = pg["lr"]
                            pg["lr"] = pg["initial_lr"] * lr_scale
                    else:
                        lr_scheduler.step()

                    optimizer.zero_grad()
                    if ema_model is not None:
                        ema_model.update()
                    grad_accum_cnt = 0
                    global_step   += 1

                    for k, v in loss_dict.items():
                        running_loss[k] = running_loss.get(k, 0) + v

                    if global_step % log_every == 0:
                        avg = {k: v / log_every for k, v in running_loss.items()}
                        logger.info(
                            f"epoch={epoch+1} step={global_step} "
                            + " ".join(f"{k}={v:.4f}" for k, v in avg.items())
                        )
                        running_loss = {}

                    if global_step % save_every == 0:
                        self._save_checkpoint(model, ema_model, global_step, output_dir)

        logger.info("학습 완료")
        self._save_checkpoint(model, ema_model, global_step, output_dir, final=True)

    def _save_checkpoint(
        self,
        model: HairS2INet,
        ema_model,
        step: int,
        output_dir: str,
        final: bool = False,
    ):
        save_dir = Path(output_dir) / (f"checkpoint-{step}" if not final else "final")
        save_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "matte_cnn":      model.matte_cnn.state_dict(),
            "sd3_controlnet": model.sd3_controlnet.state_dict(),
            "null_embeddings": {
                "hidden_states": model.null_encoder_hidden_states,
                "pooled_projections": model.null_pooled_projections,
            },
            "step":           step,
        }
        torch.save(state, save_dir / "hair_s2i_modules.pt")

        if ema_model is not None:
            with ema_model.average_parameters():
                ema_state = {
                    "matte_cnn":      model.matte_cnn.state_dict(),
                    "sd3_controlnet": model.sd3_controlnet.state_dict(),
                    "null_embeddings": {
                        "hidden_states": model.null_encoder_hidden_states,
                        "pooled_projections": model.null_pooled_projections,
                    },
                    "step":           step,
                }
            torch.save(ema_state, save_dir / "hair_s2i_modules_ema.pt")

        logger.info(f"체크포인트 저장: {save_dir} (EMA: {ema_model is not None})")
