"""
batch inference: dataset3/braid/img/test 의 N개 샘플에 대해 추론
결과: results/batch_infer/{stem}.png
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from PIL import Image
from pathlib import Path

from src.models.hair_s2i_net import HairS2INet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str,
                        default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="hair_s2i_modules.pt 또는 hair_s2i_modules_ema.pt 경로")
    parser.add_argument("--dataset_root", type=str, default="dataset3")
    parser.add_argument("--split", type=str, default="braid",
                        help="braid / unbraid")
    parser.add_argument("--subset", type=str, default="test")
    parser.add_argument("--n", type=int, default=16,
                        help="추론할 샘플 수 (0이면 전체)")
    parser.add_argument("--stems", type=str, default="",
                        help="콤마 구분 stem 목록 (예: braid_2548,braid_2562). 지정 시 --n 무시")
    parser.add_argument("--output_dir", type=str, default="results/batch_infer")
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--no_compositor", action="store_true")
    return parser.parse_args()


def load_model(args, device):
    print(f"모델 로드: {args.pretrained_model}")
    model = HairS2INet(args.pretrained_model).to(device)
    model.eval()

    print(f"체크포인트 로드: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    if "matte_cnn" in state:
        model.matte_cnn.load_state_dict(state["matte_cnn"])
        model.matte_cnn.to(torch.bfloat16)
        model.sd3_controlnet.load_state_dict(state["sd3_controlnet"])
        if "null_embeddings" in state:
            model.null_encoder_hidden_states.data.copy_(
                state["null_embeddings"]["hidden_states"].data)
            model.null_pooled_projections.data.copy_(
                state["null_embeddings"]["pooled_projections"].data)
        print(f"  step={state.get('step', 'unknown')}")
    else:
        model.load_state_dict(state, strict=False)
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.dataset_root) / args.split
    img_dir    = root / "img"    / args.subset
    sketch_dir = root / "sketch" / args.subset
    matte_dir  = root / "matte"  / args.subset
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stems:
        stems = [s.strip() for s in args.stems.split(",")]
    else:
        stems = sorted(p.stem for p in img_dir.glob("*.png"))
        if args.n > 0:
            stems = stems[:args.n]
    print(f"추론 대상: {len(stems)}개")

    model = load_model(args, device)

    for i, stem in enumerate(stems):
        print(f"[{i+1}/{len(stems)}] {stem}", end=" ... ", flush=True)

        bg  = Image.open(img_dir    / f"{stem}.png").convert("RGB")
        sk  = Image.open(sketch_dir / f"{stem}.png").convert("RGB")
        mt  = Image.open(matte_dir  / f"{stem}.png").convert("L")

        with torch.no_grad():
            result = model.inference(
                background=bg,
                sketch=sk,
                matte=mt,
                num_steps=args.num_steps,
                guidance_scale=args.guidance,
                size=(args.size, args.size),
                seed=args.seed,
                use_compositor=not args.no_compositor,
            )

        result.save(out_dir / f"{stem}.png")
        print("done")

    print(f"\n완료: {out_dir}")


if __name__ == "__main__":
    main()
