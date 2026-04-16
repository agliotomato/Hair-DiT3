"""matte-gated blending 비활성화 추론 스크립트 (디버그용)

hair-dit vs hair-dit3 색 following 차이 원인 검증:
  - matte-gated blending 제거 → hair-dit와 동일한 inference 구조
  - 색을 따르면 matte-gated blending이 원인
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from PIL import Image

from src.models.hair_s2i_net import HairS2INet


def parse_args():
    parser = argparse.ArgumentParser(description="matte-gating 제거 추론 (디버그)")
    parser.add_argument("--pretrained_model", type=str,
                        default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--background",  type=str, required=True)
    parser.add_argument("--sketch",      type=str, required=True)
    parser.add_argument("--matte",       type=str, required=True)
    parser.add_argument("--output",      type=str, default="result_no_gate.png")
    parser.add_argument("--num_steps",   type=int, default=28)
    parser.add_argument("--guidance",    type=float, default=7.0)
    parser.add_argument("--size",        type=int, default=512)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--no_compositor", action="store_true")
    return parser.parse_args()


def patch_no_matte_gate(model: HairS2INet):
    """_blend()를 matte-gating 없는 버전으로 교체"""
    def _blend_no_gate(self, residuals, matte_tokens):
        return residuals  # matte-gating 완전 제거

    import types
    model._blend = types.MethodType(_blend_no_gate, model)
    print("  [patch] matte-gated blending 비활성화")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"모델 로드: {args.pretrained_model}")
    model = HairS2INet(args.pretrained_model).to(device)
    model.eval()

    if args.checkpoint:
        print(f"체크포인트 로드: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        if "matte_cnn" in state:
            model.matte_cnn.load_state_dict(state["matte_cnn"])
            model.matte_cnn.to(torch.bfloat16)
            model.sd3_controlnet.load_state_dict(state["sd3_controlnet"])
            if "null_embeddings" in state:
                model.null_encoder_hidden_states.data.copy_(state["null_embeddings"]["hidden_states"].data)
                model.null_pooled_projections.data.copy_(state["null_embeddings"]["pooled_projections"].data)
            print(f"  Custom modules loaded (step={state.get('step', 'unknown')})")
        else:
            model.load_state_dict(state, strict=False)

    # matte-gated blending 제거 패치
    patch_no_matte_gate(model)

    background = Image.open(args.background).convert("RGB")
    sketch     = Image.open(args.sketch).convert("RGB")
    matte      = Image.open(args.matte).convert("L")

    print(f"추론 시작: steps={args.num_steps}, guidance={args.guidance}")
    result = model.inference(
        background=background,
        sketch=sketch,
        matte=matte,
        num_steps=args.num_steps,
        guidance_scale=args.guidance,
        size=(args.size, args.size),
        seed=args.seed,
        use_compositor=not args.no_compositor,
    )

    result.save(args.output)
    print(f"결과 저장: {args.output}")


if __name__ == "__main__":
    main()
