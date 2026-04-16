"""
warped_matte vs target_matte vs union + sketch coverage 시각화.
모델 추론 없이 matte/sketch만 확인.

출력: results/debug_matte_union/{group}/{ref}_on_{tgt}.png
패널: tgt_bg | warped_matte | tgt_matte | union | sketch_in_union | sketch_out_union

sketch_in_union  (노랑): sketch가 union_matte 안에 있는 영역
sketch_out_union (보라): sketch가 union_matte 밖에 있는 영역 (비어있는 union 부분)

통계:
  IoU              = warped_matte ∩ tgt_matte / warped_matte ∪ tgt_matte
  sketch_coverage  = sketch ∩ union_matte / union_matte  (union 안에서 sketch 비율)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

from scripts.hypothesis1 import get_matte_bbox, warp_sketch_to_matte

SIZE = 512

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


def to_rgb_mask(arr_01: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    out = np.zeros((arr_01.shape[0], arr_01.shape[1], 3), dtype=np.uint8)
    for c, v in enumerate(color):
        out[..., c] = (arr_01 * v).clip(0, 255).astype(np.uint8)
    return out


def sketch_to_binary(sketch_pil: Image.Image, threshold: int = 20) -> np.ndarray:
    """sketch RGB → binary [H,W] float32 (선이 있는 곳 = 1)."""
    arr = np.array(sketch_pil.convert("RGB").resize((SIZE, SIZE)), dtype=np.float32)
    # 선 = 밝은 픽셀 (배경 = 검정)
    bright = arr.max(axis=-1)  # [H,W]
    return (bright > threshold).astype(np.float32)


def make_row(tgt_bg, warped_mt_arr, tgt_mt_arr, sketch_arr, pad=4):
    """
    6개 패널:
    tgt_bg | warped_matte | tgt_matte | union | sketch_in_union | sketch_out_union
    """
    union = np.clip(warped_mt_arr + tgt_mt_arr, 0.0, 1.0)

    # sketch가 union 안에 있는 부분 (노랑)
    sketch_in  = sketch_arr * union
    # union에서 sketch가 없는 부분 (보라) — 생성은 되지만 sketch 조건 없음
    sketch_out = union * (1.0 - np.clip(sketch_arr, 0, 1))

    panels = [
        np.array(tgt_bg.convert("RGB").resize((SIZE, SIZE))),
        to_rgb_mask(warped_mt_arr),
        to_rgb_mask(tgt_mt_arr),
        to_rgb_mask(union,       color=(100, 220, 100)),  # 초록
        to_rgb_mask(sketch_in,   color=(255, 230,  50)),  # 노랑
        to_rgb_mask(sketch_out,  color=(180,  80, 220)),  # 보라
    ]

    spacer = np.full((SIZE, pad, 3), 50, dtype=np.uint8)
    row = panels[0]
    for p in panels[1:]:
        row = np.concatenate([row, spacer, p], axis=1)
    return row, union, sketch_in, sketch_out


def add_header(row_img: Image.Image, labels: list, cell_w: int, pad: int) -> Image.Image:
    header_h = 24
    header = Image.new("RGB", (row_img.width, header_h), (20, 20, 20))
    draw = ImageDraw.Draw(header)
    for i, label in enumerate(labels):
        draw.text((i * (cell_w + pad) + 4, 4), label, fill=(200, 200, 200))
    combined = Image.new("RGB", (row_img.width, header_h + row_img.height), (20, 20, 20))
    combined.paste(header, (0, 0))
    combined.paste(row_img, (0, header_h))
    return combined


def main():
    data_root  = Path("dataset3/braid")
    output_dir = Path("results/debug_matte_union")

    for group_name, group in GROUPS.items():
        ref_id = group["ref"]
        out_dir = output_dir / group_name
        out_dir.mkdir(parents=True, exist_ok=True)

        ref_sketch = Image.open(data_root / "sketch" / "test" / f"{ref_id}.png").convert("RGB")
        ref_matte  = Image.open(data_root / "matte"  / "test" / f"{ref_id}.png").convert("L").resize((SIZE, SIZE))

        for tgt_id in group["targets"]:
            tgt_bg    = Image.open(data_root / "img"   / "test" / f"{tgt_id}.png")
            tgt_matte = Image.open(data_root / "matte" / "test" / f"{tgt_id}.png").convert("L").resize((SIZE, SIZE))

            warped_matte  = warp_sketch_to_matte(ref_matte,  ref_matte, tgt_matte)
            warped_sketch = warp_sketch_to_matte(ref_sketch, ref_matte, tgt_matte)

            warped_arr = np.array(warped_matte,  dtype=np.float32) / 255.0
            tgt_arr    = np.array(tgt_matte,     dtype=np.float32) / 255.0
            sketch_arr = sketch_to_binary(warped_sketch)

            row, union, sketch_in, sketch_out = make_row(
                tgt_bg, warped_arr, tgt_arr, sketch_arr
            )
            row_img = Image.fromarray(row)

            labels = ["tgt_bg", "warped_matte", "tgt_matte",
                      "union(green)", "sketch_in(yellow)", "sketch_out(purple)"]
            row_img = add_header(row_img, labels, cell_w=SIZE, pad=4)

            # 통계
            inter     = (warped_arr * tgt_arr).sum()
            union_sum = union.sum() + 1e-6
            iou       = inter / (union_sum + 1e-6)
            coverage  = sketch_in.sum() / union_sum   # union 안에서 sketch 비율
            uncovered = sketch_out.sum() / union_sum  # union 안에서 sketch 없는 비율

            draw = ImageDraw.Draw(row_img)
            draw.text(
                (4, row_img.height - 18),
                f"{ref_id} -> {tgt_id}  IoU={iou:.3f}  "
                f"sketch_in_union={coverage:.1%}  uncovered={uncovered:.1%}",
                fill=(255, 220, 80),
            )

            out_path = out_dir / f"{ref_id}_on_{tgt_id}.png"
            row_img.save(out_path)
            print(
                f"[{group_name}] {ref_id} -> {tgt_id}  "
                f"IoU={iou:.3f}  sketch_coverage={coverage:.1%}  uncovered={uncovered:.1%}"
            )

    print("\n완료.")


if __name__ == "__main__":
    main()
