"""
Hypothesis 1: Face Context Dependency 실험

sketch + matte + 위치를 완전히 고정하고, 얼굴(face)만 변경.
DiT global attention이 face context를 hair 생성에 반영하는지 검증.

Fixed   : ref_sketch, ref_matte (warp 없음, 위치 고정), seed
Varying : background = tgt_img * (1 - ref_matte)
          → hair 구멍(ref_matte=1)은 검정, 얼굴+배경은 target 사람 픽셀

이렇게 하면 모델이 보는 얼굴만 바뀌고, hair 생성 조건(sketch/matte)은 동일.
출력 = hair_only (ref_matte로 마스킹).
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from PIL import Image, ImageFilter
from pathlib import Path

from src.models.hair_s2i_net import HairS2INet


def remove_hair_region(bg_pil: Image.Image, matte_pil: Image.Image) -> Image.Image:
    """
    matte 영역(hair)을 검정(0)으로 마스킹하여 반환.
    훈련 시 background = img * (1 - matte) 이므로 동일한 분포를 맞춤.
    """
    bg  = bg_pil.convert("RGB").resize((512, 512))
    mt  = matte_pil.convert("L").resize((512, 512))

    bg_arr = np.array(bg, dtype=np.float32)
    mt_arr = np.array(mt, dtype=np.float32) / 255.0  # [0,1]

    # hair 영역 = 0 (검정), 배경 = 원본
    result = bg_arr * (1.0 - mt_arr[..., None])
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


GROUPS = {
    "back_2534": {
        "ref":     "braid_2534",  # sketch + matte 고정 (warp 없음)
        "targets": ["braid_2534", "braid_2572", "braid_2574",
                    "braid_2576", "braid_2592", "braid_2617"],
    },
    "back_2562": {
        "ref":     "braid_2562",
        "targets": ["braid_2562", "braid_2572", "braid_2574",
                    "braid_2576", "braid_2592", "braid_2617"],
    },
    "nanobana_2537": {
        "ref":      "braid_2537",          # sketch + matte 고정
        "face_dir": "dataset/nanobanana",
        "targets":  [
            "braid_2537_smile",   # 웃는 얼굴
            "braid_2537_sad",     # 슬픈 얼굴
        ],
    },
    "nanobana_2562": {
        "ref":      "braid_2562",
        "face_dir": "dataset/nanobanana",
        "targets":  [
            "braid_2562_smile",
            "braid_2562_sad",
        ],
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

    fillcolor = 0 if sketch_pil.mode == "L" else (0, 0, 0)
    warped = sketch_pil.transform(
        (W, H),
        Image.AFFINE,
        coeffs,
        resample=Image.BILINEAR,
        fillcolor=fillcolor,
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
    parser.add_argument("--guidance",          type=float, default=7.5)
    parser.add_argument("--controlnet_scale", type=float, default=1.0)
    parser.add_argument("--size",             type=int,   default=512)
    parser.add_argument("--seed",       type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    model = load_model(args.pretrained_model, args.checkpoint, device)

    for group_name, group in GROUPS.items():
        ref_id   = group["ref"]
        face_dir = Path(group["face_dir"]) if "face_dir" in group else None
        out_group = output_dir / group_name
        out_group.mkdir(parents=True, exist_ok=True)

        ref_sketch = Image.open(data_root / "sketch" / "test" / f"{ref_id}.png").convert("RGB")
        ref_matte  = Image.open(data_root / "matte"  / "test" / f"{ref_id}.png").convert("L")
        ref_mt_arr = np.array(ref_matte.convert("L").resize((args.size, args.size)),
                              dtype=np.float32) / 255.0

        grid_imgs = []

        # baseline: ref 본인 이미지로 추론
        baseline_path = out_group / f"{ref_id}_full_{ref_id}.png"
        baseline_hair_path = out_group / f"{ref_id}_bg_{ref_id}.png"
        if baseline_path.exists():
            print(f"[skip baseline] {baseline_path}")
            baseline_full = Image.open(baseline_path).convert("RGB")
            baseline_hair = Image.open(baseline_hair_path).convert("RGB")
        else:
            ref_img = Image.open(data_root / "img" / "test" / f"{ref_id}.png").convert("RGB")
            print(f"[{group_name}] baseline: ref={ref_id}")
            baseline_result = model.inference(
                background=ref_img,
                sketch=ref_sketch,
                matte=ref_matte,
                num_steps=args.num_steps,
                guidance_scale=args.guidance,
                controlnet_scale=args.controlnet_scale,
                size=(args.size, args.size),
                seed=args.seed,
            )
            baseline_arr  = np.array(baseline_result.resize((args.size, args.size)), dtype=np.float32)
            baseline_hair_arr = (baseline_arr * ref_mt_arr[..., None]).clip(0, 255).astype(np.uint8)
            baseline_full = baseline_result.resize((args.size, args.size))
            baseline_hair = Image.fromarray(baseline_hair_arr)
            baseline_full.save(baseline_path)
            baseline_hair.save(baseline_hair_path)
            print(f"  저장: {baseline_path}")
        grid_imgs.append((f"{ref_id}(원본)", baseline_full, baseline_hair))

        for tgt_name in group["targets"]:
            out_path = out_group / f"{ref_id}_bg_{tgt_name}.png"
            if out_path.exists():
                print(f"[skip] {out_path}")
                full_path = out_group / f"{ref_id}_full_{tgt_name}.png"
                grid_imgs.append((
                    tgt_name,
                    Image.open(full_path).convert("RGB") if full_path.exists() else Image.open(out_path).convert("RGB"),
                    Image.open(out_path).convert("RGB"),
                ))
                continue

            # 나노바나나 그룹이면 face_dir에서, 아니면 braid img/test에서 로드
            if face_dir is not None:
                tgt_img = Image.open(face_dir / f"{tgt_name}.png").convert("RGB")
            else:
                tgt_img = Image.open(data_root / "img" / "test" / f"{tgt_name}.png").convert("RGB")

            print(f"[{group_name}] ref={ref_id}  face={tgt_name}")
            result = model.inference(
                background=tgt_img,
                sketch=ref_sketch,
                matte=ref_matte,
                num_steps=args.num_steps,
                guidance_scale=args.guidance,
                controlnet_scale=args.controlnet_scale,
                size=(args.size, args.size),
                seed=args.seed,
            )

            result_arr = np.array(result.resize((args.size, args.size)), dtype=np.float32)
            hair_only  = (result_arr * ref_mt_arr[..., None]).clip(0, 255).astype(np.uint8)
            hair_img   = Image.fromarray(hair_only)

            # hair_only 저장
            hair_img.save(out_path)
            # full 결과 저장
            full_path  = out_group / f"{ref_id}_full_{tgt_name}.png"
            result.resize((args.size, args.size)).save(full_path)
            print(f"  저장: {out_path}, {full_path}")
            grid_imgs.append((tgt_name, result.resize((args.size, args.size)), hair_img))

        # 비교 그리드: 각 표정마다 [full | hair_only] 2열
        if grid_imgs:
            from PIL import ImageDraw
            label_h  = 20
            cell_w   = args.size
            n        = len(grid_imgs)
            # 열: 표정 수 × 2 (full + hair)
            grid_w   = (cell_w * 2 + 4) * n + 4 * (n - 1)
            grid     = Image.new("RGB", (grid_w, cell_w + label_h), (20, 20, 20))
            draw     = ImageDraw.Draw(grid)
            for i, (name, full_img, hair_img) in enumerate(grid_imgs):
                x = i * (cell_w * 2 + 8)
                draw.text((x + 4,           2), f"{name} full", fill=(180, 180, 180))
                draw.text((x + cell_w + 8,  2), f"{name} hair", fill=(100, 220, 100))
                grid.paste(full_img, (x,           label_h))
                grid.paste(hair_img, (x + cell_w + 4, label_h))
            grid.save(out_group / "comparison_grid.png")
            print(f"  그리드 저장: {out_group}/comparison_grid.png")

    print("완료.")


if __name__ == "__main__":
    main()
