"""단일 이미지 추론 스크립트"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from PIL import Image
from pathlib import Path

from src.models.hair_s2i_net import HairS2INet


def parse_args():
    parser = argparse.ArgumentParser(description="Hair Sketch-to-Image 추론")
    parser.add_argument("--pretrained_model", type=str,
                        default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="hair_s2i_modules.pt 경로")
    parser.add_argument("--background",  type=str, required=True)
    parser.add_argument("--sketch",      type=str, required=True)
    parser.add_argument("--matte",       type=str, required=True)
    parser.add_argument("--prompt",      type=str, default="")
    parser.add_argument("--output",      type=str, default="result.png")
    parser.add_argument("--num_steps",   type=int, default=28)
    parser.add_argument("--guidance",    type=float, default=7.0)
    parser.add_argument("--size",        type=int, default=512)
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"모델 로드: {args.pretrained_model}")
    model = HairS2INet(args.pretrained_model).to(device)
    model.eval()

    # 체크포인트 로드 (신규 모듈만)
    print(f"체크포인트 로드: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    model.sketch_encoder.load_state_dict(state["sketch_encoder"])
    model.latent_gate.load_state_dict(state["latent_gate"])
    model.hair_controlnet.load_state_dict(state["hair_controlnet"])
    print(f"  step={state.get('step', 'unknown')}")

    # 입력 이미지 로드
    background = Image.open(args.background).convert("RGB")
    sketch     = Image.open(args.sketch).convert("L")
    matte      = Image.open(args.matte).convert("L")

    print(f"추론 시작: prompt='{args.prompt}', steps={args.num_steps}, guidance={args.guidance}")
    result = model.inference(
        background=background,
        sketch=sketch,
        matte=matte,
        prompt=args.prompt,
        num_steps=args.num_steps,
        guidance_scale=args.guidance,
        size=(args.size, args.size),
        seed=args.seed,
    )

    result.save(args.output)
    print(f"결과 저장: {args.output}")


if __name__ == "__main__":
    main()
