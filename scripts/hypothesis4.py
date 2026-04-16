"""
Hypothesis 4: Union Matte로 헤어 생성

hypothesis1과 동일한 셋업이지만,
matte = warped_matte ∪ tgt_matte 로 생성 영역을 확장.

목적: warped_matte만 사용할 때 target 기존 머리가 배경에서 삐져나오는 문제 해소.

출력: results/hypothesis4/{group}/{ref}_on_{tgt}.png
      + side_by_side: hypothesis1 결과 vs hypothesis4 결과
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


def make_union_matte(warped_matte: Image.Image, tgt_matte: Image.Image) -> Image.Image:
    """warped_matte ∪ tgt_matte → PIL L 이미지."""
    w_arr = np.array(warped_matte.convert("L"), dtype=np.float32) / 255.0
    t_arr = np.array(tgt_matte.convert("L"),   dtype=np.float32) / 255.0
    union = np.clip(w_arr + t_arr, 0.0, 1.0)
    return Image.fromarray((union * 255).astype(np.uint8), mode="L")


def make_side_by_side(h1_path: Path, h4_path: Path, label_h=20) -> Image.Image:
    """hypothesis1 결과 | hypothesis4 결과 비교 이미지."""
    from PIL import ImageDraw
    img1 = Image.open(h1_path).convert("RGB").resize((SIZE, SIZE)) if h1_path.exists() else Image.new("RGB", (SIZE, SIZE), (80, 80, 80))
    img4 = Image.open(h4_path).convert("RGB").resize((SIZE, SIZE))

    out = Image.new("RGB", (SIZE * 2 + 4, SIZE + label_h), (20, 20, 20))
    draw = ImageDraw.Draw(out)
    draw.text((4,       2), "hypo1 (warped_matte)", fill=(180, 180, 180))
    draw.text((SIZE + 8, 2), "hypo4 (union_matte)",  fill=(100, 220, 100))
    out.paste(img1, (0,        label_h))
    out.paste(img4, (SIZE + 4, label_h))
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
    parser = argparse.ArgumentParser(description="Hypothesis 4: Union Matte 실험")
    parser.add_argument("--pretrained_model", type=str,
                        default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--checkpoint",       type=str, required=True)
    parser.add_argument("--data_root",        type=str, default="dataset3/braid")
    parser.add_argument("--output_dir",       type=str, default="results/hypothesis4")
    parser.add_argument("--hypo1_dir",        type=str, default="results/hypothesis1",
                        help="hypothesis1 결과 디렉토리 (side-by-side 비교용)")
    parser.add_argument("--num_steps",        type=int,   default=28)
    parser.add_argument("--guidance",         type=float, default=7.5)
    parser.add_argument("--controlnet_scale", type=float, default=1.0)
    parser.add_argument("--seed",             type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

            warped_sketch = warp_sketch_to_matte(ref_sketch, ref_matte, tgt_matte)
            warped_matte  = warp_sketch_to_matte(ref_matte,  ref_matte, tgt_matte)
            union_matte   = make_union_matte(warped_matte, tgt_matte)

            if out_path.exists():
                print(f"[skip] {out_path}")
                result = Image.open(out_path).convert("RGB")
            else:
                print(f"[{group_name}] {ref_id} → {tgt_id} (union matte)")
                result = model.inference(
                    background=tgt_bg,
                    sketch=warped_sketch,
                    matte=union_matte,
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
            cmp = make_side_by_side(h1_path, out_path)
            cmp.save(cmp_path)

    print("\n완료.")


if __name__ == "__main__":
    main()
