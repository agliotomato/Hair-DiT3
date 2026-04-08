"""affine warp 시각화 디버그 스크립트"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

from scripts.hypothesis1 import get_matte_bbox, warp_sketch_to_matte

DATA = Path("dataset/braid")
OUT  = Path("results/debug_affine")
OUT.mkdir(parents=True, exist_ok=True)

REF_ID = "braid_2534"
TARGETS = ["braid_2574", "braid_2576", "braid_2592", "braid_2617"]

ref_sketch = Image.open(DATA / "sketch" / "test" / f"{REF_ID}.png").convert("L")
ref_matte  = Image.open(DATA / "matte"  / "test" / f"{REF_ID}.png").convert("L")


def draw_bbox(img_pil, bbox, color="red", width=3):
    """이미지 위에 bbox 사각형 그리기 (RGB 복사본 반환)"""
    vis = img_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    x0, y0, x1, y1 = bbox
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return vis


def overlay_sketch_on_bg(bg_pil, sketch_pil, matte_pil, alpha=0.6):
    """sketch를 빨간 채널로 배경 위에 오버레이"""
    bg   = bg_pil.convert("RGB").resize((512, 512))
    sk   = sketch_pil.convert("L").resize((512, 512))
    sk_arr = np.array(sk)
    bg_arr = np.array(bg)

    # sketch 선이 있는 곳(밝은 부분)을 빨간색으로 표시
    mask = sk_arr > 30
    overlay = bg_arr.copy()
    overlay[mask] = [255, 50, 50]

    # matte 경계선 초록색
    mt_arr = np.array(matte_pil.convert("L").resize((512, 512)))
    from scipy.ndimage import binary_erosion
    boundary = mt_arr > 10
    eroded   = binary_erosion(boundary, iterations=3)
    edge     = boundary & ~eroded
    overlay[edge] = [50, 255, 50]

    return Image.fromarray(
        (bg_arr * (1 - alpha) + overlay * alpha).clip(0, 255).astype(np.uint8)
    )


for tgt_id in TARGETS:
    tgt_matte = Image.open(DATA / "matte" / "test" / f"{tgt_id}.png").convert("L")
    tgt_bg    = Image.open(DATA / "img"   / "test" / f"{tgt_id}.png").convert("RGB")

    warped = warp_sketch_to_matte(ref_sketch, ref_matte, tgt_matte)

    src_bbox = get_matte_bbox(ref_matte)
    tgt_bbox = get_matte_bbox(tgt_matte)

    # 열 1: ref sketch + ref matte bbox
    col1 = draw_bbox(ref_sketch.convert("RGB"), src_bbox, color="blue")
    # 열 2: warped sketch + target matte bbox
    col2 = draw_bbox(warped.convert("RGB"), tgt_bbox, color="red")
    # 열 3: warped sketch를 target 배경에 오버레이
    col3 = overlay_sketch_on_bg(tgt_bg, warped, tgt_matte)

    W, H = 512, 512
    canvas = Image.new("RGB", (W * 3, H))
    canvas.paste(col1.resize((W, H)), (0,   0))
    canvas.paste(col2.resize((W, H)), (W,   0))
    canvas.paste(col3.resize((W, H)), (W*2, 0))

    out_path = OUT / f"{REF_ID}_on_{tgt_id}_affine_debug.png"
    canvas.save(out_path)
    print(f"저장: {out_path}")
    print(f"  src bbox: {src_bbox}")
    print(f"  tgt bbox: {tgt_bbox}")

print("완료. results/debug_affine/ 확인하세요.")
