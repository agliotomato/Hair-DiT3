"""
Hypothesis 1: Face Context Dependency 실험

동일한 reference sketch를 여러 다른 인물에게 적용.
sketch를 target matte bounding box에 affine warp하여 정렬 후 추론.

실험 그룹:
  back : braid_2534(ref), 2572, 2574, 2576, 2592, 2617
  side : braid_2537(ref), 2539, 2676
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from PIL import Image
from pathlib import Path

from src.models.hair_s2i_net import HairS2INet


GROUPS = {
    "back": {
        "ref":     "braid_2534",
        "targets": ["braid_2534", "braid_2572", "braid_2574",
                    "braid_2576", "braid_2592", "braid_2617"],
    },
    "side": {
        "ref":     "braid_2537",
        "targets": ["braid_2537", "braid_2539", "braid_2625", "braid_2676"],
    },
}


def get_matte_bbox(matte_pil: Image.Image, threshold: int = 10):
    """matte에서 foreground bounding box 반환 (x0, y0, x1, y1)."""
    arr = np.array(matte_pil.convert("L"))
    ys, xs = np.where(arr > threshold)
    if len(xs) == 0:
        w, h = matte_pil.size
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def warp_sketch_to_matte(
    sketch_pil: Image.Image,
    src_matte_pil: Image.Image,
    tgt_matte_pil: Image.Image,
) -> Image.Image:
    """
    sketch를 src_matte bbox → tgt_matte bbox로 affine warp.

    PIL AFFINE transform은 역방향(출력→입력) 매핑을 받으므로:
        src_x = (tgt_x - tx) / sx
        src_y = (tgt_y - ty) / sy
    coeffs = (1/sx, 0, -tx/sx, 0, 1/sy, -ty/sy)
    """
    W, H = sketch_pil.size

    sx0, sy0, sx1, sy1 = get_matte_bbox(src_matte_pil)
    tx0, ty0, tx1, ty1 = get_matte_bbox(tgt_matte_pil)

    src_w, src_h = sx1 - sx0, sy1 - sy0
    tgt_w, tgt_h = tx1 - tx0, ty1 - ty0

    if src_w <= 0 or src_h <= 0 or tgt_w <= 0 or tgt_h <= 0:
        return sketch_pil

    scale_x = tgt_w / src_w
    scale_y = tgt_h / src_h

    # tgt bbox 좌상단이 src bbox 좌상단에 매핑되도록 translation
    tx = tx0 - sx0 * scale_x
    ty = ty0 - sy0 * scale_y

    # PIL AFFINE 역변환 coefficients
    coeffs = (1 / scale_x, 0, -tx / scale_x,
              0, 1 / scale_y, -ty / scale_y)

    warped = sketch_pil.transform(
        (W, H),
        Image.AFFINE,
        coeffs,
        resample=Image.BILINEAR,
        fillcolor=0,
    )
    return warped


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
    parser = argparse.ArgumentParser(description="Hypothesis 1: Context Dependency 실험")
    parser.add_argument("--pretrained_model", type=str,
                        default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root",  type=str, default="dataset/braid")
    parser.add_argument("--output_dir", type=str, default="results/hypothesis1")
    parser.add_argument("--num_steps",  type=int,   default=28)
    parser.add_argument("--guidance",   type=float, default=7.5)
    parser.add_argument("--size",       type=int,   default=512)
    parser.add_argument("--seed",       type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    model = load_model(args.pretrained_model, args.checkpoint, device)

    for group_name, group in GROUPS.items():
        ref_id = group["ref"]
        out_group = output_dir / group_name
        out_group.mkdir(parents=True, exist_ok=True)

        ref_sketch = Image.open(data_root / "sketch" / "test" / f"{ref_id}.png").convert("L")
        ref_matte  = Image.open(data_root / "matte"  / "test" / f"{ref_id}.png").convert("L")

        for tgt_id in group["targets"]:
            out_path = out_group / f"{ref_id}_on_{tgt_id}.png"
            if out_path.exists():
                print(f"[skip] {out_path}")
                continue

            tgt_bg    = Image.open(data_root / "img"    / "test" / f"{tgt_id}.png").convert("RGB")
            tgt_matte = Image.open(data_root / "matte"  / "test" / f"{tgt_id}.png").convert("L")

            warped_sketch = warp_sketch_to_matte(ref_sketch, ref_matte, tgt_matte)

            print(f"[{group_name}] {ref_id} → {tgt_id}")
            result = model.inference(
                background=tgt_bg,
                sketch=warped_sketch,
                matte=tgt_matte,
                num_steps=args.num_steps,
                guidance_scale=args.guidance,
                size=(args.size, args.size),
                seed=args.seed,
            )
            result.save(out_path)
            print(f"  저장: {out_path}")

    print("완료.")


if __name__ == "__main__":
    main()
