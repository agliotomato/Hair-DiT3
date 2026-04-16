"""
Hypothesis 5: tgt_bg 머리 제거 후 warped_matte만 사용

hypothesis1 문제: tgt_bg에 기존 머리가 포함되어 있어
warped_matte 밖 tgt_matte 영역에서 기존 머리가 compositor를 통해 노출됨.

해결: tgt_bg에서 tgt_matte 영역을 cv2.inpaint로 제거 후 사용.
matte는 union 없이 warped_matte만 사용.

비교: results/hypothesis5/{group}/{ref}_on_{tgt}.png
      results/hypothesis5/{group}_compare/{ref}_on_{tgt}.png (hypo1 vs hypo5)
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from src.models.hair_s2i_net import HairS2INet
from scripts.hypothesis1 import get_matte_bbox, warp_sketch_to_matte


GROUPS = {
    "back": {
        "ref":     "braid_2534",
        "targets": ["braid_2534", "braid_2572", "braid_2574",
                    "braid_2576", "braid_2592", "braid_2617"],
    },
    "back_2562": {
        "ref":     "braid_2562",
        "targets": ["braid_2562", "braid_2572", "braid_2574",
                    "braid_2576", "braid_2592", "braid_2617"],
    },
}

SIZE = 512


def erase_hair(bg_pil: Image.Image, matte_pil: Image.Image,
               inpaint_radius: int = 15) -> Image.Image:
    """
    tgt_matte 영역을 cv2.inpaint(TELEA)로 제거.
    주변 픽셀에서 자연스럽게 채워넣음.

    Args:
        bg_pil: target 원본 이미지 (RGB)
        matte_pil: target hair matte (L)
        inpaint_radius: inpaint 탐색 반경 (클수록 넓게 보지만 느림)
    """
    bg  = np.array(bg_pil.convert("RGB").resize((SIZE, SIZE)))
    mt  = np.array(matte_pil.convert("L").resize((SIZE, SIZE)))

    # inpaint mask: 255 = 채울 영역 (hair), 0 = 유지
    inpaint_mask = (mt > 20).astype(np.uint8) * 255

    # cv2는 BGR
    bg_bgr    = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
    result_bgr = cv2.inpaint(bg_bgr, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


def make_side_by_side(h1_path: Path, h5_path: Path) -> Image.Image:
    from PIL import ImageDraw
    img1 = Image.open(h1_path).convert("RGB").resize((SIZE, SIZE)) if h1_path.exists() \
           else Image.new("RGB", (SIZE, SIZE), (60, 60, 60))
    img5 = Image.open(h5_path).convert("RGB").resize((SIZE, SIZE))

    label_h = 20
    out = Image.new("RGB", (SIZE * 2 + 4, SIZE + label_h), (20, 20, 20))
    draw = ImageDraw.Draw(out)
    draw.text((4,        2), "hypo1 (raw tgt_bg)",    fill=(180, 180, 180))
    draw.text((SIZE + 8, 2), "hypo5 (inpaint tgt_bg)", fill=(100, 220, 100))
    out.paste(img1, (0,        label_h))
    out.paste(img5, (SIZE + 4, label_h))
    return out


def load_model(pretrained_model, checkpoint, device):
    print(f"모델 로드: {pretrained_model}")
    model = HairS2INet(pretrained_model).to(device)
    model.eval()

    if checkpoint:
        print(f"체크포인트 로드: {checkpoint}")
        state = torch.load(checkpoint, map_location=device)
        if "matte_cnn" in state:
            model.matte_cnn.load_state_dict(state["matte_cnn"])
            model.matte_cnn.to(torch.bfloat16)
            model.sd3_controlnet.load_state_dict(state["sd3_controlnet"])
            if "null_embeddings" in state:
                model.null_encoder_hidden_states.data.copy_(
                    state["null_embeddings"]["hidden_states"].data)
                model.null_pooled_projections.data.copy_(
                    state["null_embeddings"]["pooled_projections"].data)
            print(f"  Custom modules loaded (step={state.get('step', 'unknown')})")
        else:
            model.load_state_dict(state, strict=False)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Hypothesis 5: tgt_bg inpaint + warped_matte")
    parser.add_argument("--pretrained_model", type=str,
                        default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--checkpoint",       type=str, required=True)
    parser.add_argument("--data_root",        type=str, default="dataset3/braid")
    parser.add_argument("--output_dir",       type=str, default="results/hypothesis5")
    parser.add_argument("--hypo1_dir",        type=str, default="results/hypothesis1")
    parser.add_argument("--inpaint_radius",   type=int, default=15)
    parser.add_argument("--num_steps",        type=int,   default=28)
    parser.add_argument("--guidance",         type=float, default=7.5)
    parser.add_argument("--controlnet_scale", type=float, default=1.0)
    parser.add_argument("--seed",             type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    hypo1_dir  = Path(args.hypo1_dir)

    model = load_model(args.pretrained_model, args.checkpoint, device)

    for group_name, group in GROUPS.items():
        ref_id  = group["ref"]
        out_grp = output_dir / group_name
        cmp_grp = output_dir / f"{group_name}_compare"
        out_grp.mkdir(parents=True, exist_ok=True)
        cmp_grp.mkdir(parents=True, exist_ok=True)

        ref_sketch = Image.open(data_root / "sketch" / "test" / f"{ref_id}.png").convert("RGB")
        ref_matte  = Image.open(data_root / "matte"  / "test" / f"{ref_id}.png").convert("L")

        for tgt_id in group["targets"]:
            out_path = out_grp / f"{ref_id}_on_{tgt_id}.png"
            cmp_path = cmp_grp / f"{ref_id}_on_{tgt_id}.png"

            tgt_bg    = Image.open(data_root / "img"   / "test" / f"{tgt_id}.png").convert("RGB")
            tgt_matte = Image.open(data_root / "matte" / "test" / f"{tgt_id}.png").convert("L")

            # warped sketch + matte (hypothesis1과 동일)
            warped_sketch = warp_sketch_to_matte(ref_sketch, ref_matte, tgt_matte)
            warped_matte  = warp_sketch_to_matte(ref_matte,  ref_matte, tgt_matte)

            # tgt_bg에서 기존 머리 inpaint 제거
            tgt_bg_clean = erase_hair(tgt_bg, tgt_matte, args.inpaint_radius)

            if out_path.exists():
                print(f"[skip] {out_path}")
            else:
                print(f"[{group_name}] {ref_id} → {tgt_id} (inpaint bg)")
                result = model.inference(
                    background=tgt_bg_clean,
                    sketch=warped_sketch,
                    matte=warped_matte,
                    num_steps=args.num_steps,
                    guidance_scale=args.guidance,
                    controlnet_scale=args.controlnet_scale,
                    size=(SIZE, SIZE),
                    seed=args.seed,
                )
                result.save(out_path)
                print(f"  저장: {out_path}")

            # side-by-side 비교
            h1_path = hypo1_dir / group_name / f"{ref_id}_on_{tgt_id}.png"
            make_side_by_side(h1_path, out_path).save(cmp_path)

    print("\n완료.")


if __name__ == "__main__":
    main()
