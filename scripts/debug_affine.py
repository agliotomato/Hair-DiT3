"""affine warp 시각화 디버그 스크립트"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import binary_erosion

from scripts.hypothesis1 import get_matte_bbox, warp_sketch_to_matte

DATA = Path("dataset3/braid")
OUT  = Path("results/debug_affine")
OUT.mkdir(parents=True, exist_ok=True)

REF_ID  = "braid_2534"
TARGETS = ["braid_2572", "braid_2574", "braid_2576", "braid_2592", "braid_2617"]

ref_sketch = Image.open(DATA / "sketch" / "test" / f"{REF_ID}.png").convert("RGB")
ref_matte  = Image.open(DATA / "matte"  / "test" / f"{REF_ID}.png").convert("L")


def draw_bbox(img_pil, bbox, color="red", width=3):
    vis = img_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    x0, y0, x1, y1 = bbox
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return vis


def matte_edge(matte_pil, size=(512, 512)):
    arr = np.array(matte_pil.convert("L").resize(size))
    boundary = arr > 10
    eroded   = binary_erosion(boundary, iterations=3)
    return boundary & ~eroded


def overlay(bg_pil, sketch_pil, matte_pil, size=(512, 512)):
    """배경 위에 sketch(빨강) + matte 경계(초록) 오버레이"""
    bg_arr = np.array(bg_pil.convert("RGB").resize(size), dtype=float)
    sk_arr = np.array(sketch_pil.convert("L").resize(size))
    out    = bg_arr.copy()
    out[sk_arr > 30]           = [255, 50,  50]   # sketch 선 → 빨강
    out[matte_edge(matte_pil)] = [50,  255, 50]   # matte 경계 → 초록
    return Image.fromarray((bg_arr * 0.4 + out * 0.6).clip(0, 255).astype(np.uint8))


W, H = 512, 512

for tgt_id in TARGETS:
    tgt_matte  = Image.open(DATA / "matte"  / "test" / f"{tgt_id}.png").convert("L")
    tgt_bg     = Image.open(DATA / "img"    / "test" / f"{tgt_id}.png").convert("RGB")

    warped_sketch = warp_sketch_to_matte(ref_sketch, ref_matte, tgt_matte)
    warped_matte  = warp_sketch_to_matte(ref_matte,  ref_matte, tgt_matte)

    src_bbox = get_matte_bbox(ref_matte)
    tgt_bbox = get_matte_bbox(tgt_matte)

    # 열 1: ref sketch + ref matte bbox (파랑)
    col1 = draw_bbox(ref_sketch, src_bbox, color="blue")
    # 열 2: warped sketch + warped matte 경계 (초록) + target matte bbox (빨강)
    col2 = overlay(warped_sketch, warped_sketch, warped_matte)
    col2 = draw_bbox(col2, tgt_bbox, color="red")
    # 열 3: target 배경 위에 warped sketch(빨강) + warped matte 경계(초록)
    col3 = overlay(tgt_bg, warped_sketch, warped_matte)

    canvas = Image.new("RGB", (W * 3, H))
    canvas.paste(col1.resize((W, H)), (0,    0))
    canvas.paste(col2.resize((W, H)), (W,    0))
    canvas.paste(col3.resize((W, H)), (W*2,  0))

    out_path = OUT / f"{REF_ID}_on_{tgt_id}.png"
    canvas.save(out_path)
    print(f"저장: {out_path}  |  src_bbox={src_bbox}  tgt_bbox={tgt_bbox}")

print("완료. results/debug_affine/ 확인하세요.")
print("열 1: ref sketch + bbox | 열 2: warped sketch + warped matte 경계 | 열 3: target 배경 오버레이")
