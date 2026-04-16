"""
Hypothesis 3: Compositor 변형 실험

동일한 입력에서 compositor 옵션만 바꿔 출력 차이 비교.

실험 조합 (noise_mode × blur_schedule):
  1. fixed  + linear   ← 현재 기본값
  2. random + linear   ← noise 고정 여부 효과
  3. fixed  + cosine   ← blur schedule 효과
  4. random + cosine   ← 두 변경 동시 적용

각 reference마다 4조건 × comparison_grid 생성.
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from PIL import Image
from pathlib import Path

from src.models.hair_s2i_net import HairS2INet


REFERENCES = ["braid_2534", "braid_2562", "braid_2574", "braid_2653"]

VARIANTS = [
    ("fixed_linear",   "fixed",  "linear"),
    ("random_linear",  "random", "linear"),
    ("fixed_cosine",   "fixed",  "cosine"),
    ("random_cosine",  "random", "cosine"),
]


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


def make_comparison_grid(rows: list) -> Image.Image:
    """rows: [(label, result_img), ...]"""
    from PIL import ImageDraw
    label_w = 160
    cell_w, cell_h = 512, 512
    total_w = label_w + cell_w
    total_h = (cell_h + 2) * len(rows)

    grid = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    for i, (label, img) in enumerate(rows):
        y = i * (cell_h + 2)
        draw.text((4, y + cell_h // 2 - 6), label, fill=(220, 220, 220))
        grid.paste(img.resize((cell_w, cell_h)), (label_w, y))

    return grid


def parse_args():
    parser = argparse.ArgumentParser(description="Hypothesis 3: Compositor 변형 실험")
    parser.add_argument("--pretrained_model", type=str,
                        default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--checkpoint",       type=str, required=True)
    parser.add_argument("--data_root",        type=str, default="dataset3/braid")
    parser.add_argument("--output_dir",       type=str, default="results/hypothesis3")
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

    model = load_model(args.pretrained_model, args.checkpoint, device)

    for ref_id in REFERENCES:
        print(f"\n======== {ref_id} ========")
        ref_dir = output_dir / ref_id
        ref_dir.mkdir(parents=True, exist_ok=True)

        face   = Image.open(data_root / "img"    / "test" / f"{ref_id}.png").convert("RGB")
        sketch = Image.open(data_root / "sketch" / "test" / f"{ref_id}.png").convert("RGB")
        matte  = Image.open(data_root / "matte"  / "test" / f"{ref_id}.png").convert("L")

        grid_rows = []

        for name, noise_mode, blur_schedule in VARIANTS:
            out_path = ref_dir / f"{name}.png"

            if out_path.exists():
                print(f"[skip] {out_path}")
                result = Image.open(out_path).convert("RGB")
            else:
                print(f"[{ref_id}] {name} 추론 (noise={noise_mode}, blur={blur_schedule}) ...")
                result = model.inference(
                    background=face,
                    sketch=sketch,
                    matte=matte,
                    num_steps=args.num_steps,
                    guidance_scale=args.guidance,
                    controlnet_scale=args.controlnet_scale,
                    size=(512, 512),
                    seed=args.seed,
                    use_compositor=True,
                    noise_mode=noise_mode,
                    blur_schedule=blur_schedule,
                )
                result.save(out_path)
                print(f"  저장: {out_path}")

            grid_rows.append((name, result))

        grid = make_comparison_grid(grid_rows)
        grid_path = ref_dir / "comparison_grid.png"
        grid.save(grid_path)
        print(f"  그리드 저장: {grid_path}")

    print("\n완료.")


if __name__ == "__main__":
    main()
