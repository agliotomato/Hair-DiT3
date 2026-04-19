"""
SketchHairSalon / HairClipV2 / Ours 3-method 정량 비교

사용법:
  python scripts/eval_compare.py
  python scripts/eval_compare.py --output eval_compare.json
  python scripts/eval_compare.py --no_fid
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
from pathlib import Path
from scripts.evaluate import (
    evaluate_single,
    load_image_np, load_matte_np,
    InceptionFeatureExtractor,
    compute_fid, _crop_hair, _crop_boundary,
)

STEMS = [
    "braid_2548", "braid_2562", "braid_2574", "braid_2590",
    "braid_2592", "braid_2617", "braid_2625", "braid_2652",
]

GT_DIR     = Path("dataset3/braid/img/test")
MATTE_DIR  = Path("dataset3/braid/matte/test")
SKETCH_DIR = Path("dataset3/braid/sketch/test")

METHODS = {
    "SketchHairSalon": {
        "pred_dir": Path("results/hairsalon"),
        "suffix": "",
    },
    "HairClipV2": {
        "pred_dir": Path("results/hairclipv2"),
        "suffix": "_result",
    },
    "Ours": {
        "pred_dir": Path("results/batch_infer_final2"),
        "suffix": "",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default="eval_compare.json")
    p.add_argument("--no_fid", action="store_true")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def mean_std(vals):
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def main():
    args = parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    inception = None
    if not args.no_fid:
        try:
            print(f"InceptionV3 로드 ({device})...")
            inception = InceptionFeatureExtractor(device=device)
        except Exception as e:
            print(f"[경고] InceptionV3 로드 실패 → FID 생략: {e}")

    all_results = {}

    for method_name, cfg in METHODS.items():
        pred_dir = cfg["pred_dir"]
        suffix   = cfg["suffix"]
        per_image = []
        feats_pred_hair, feats_gt_hair = [], []
        feats_pred_bnd,  feats_gt_bnd  = [], []

        for stem in STEMS:
            pred_path   = pred_dir / f"{stem}{suffix}.png"
            gt_path     = GT_DIR / f"{stem}.png"
            matte_path  = MATTE_DIR / f"{stem}.png"
            sketch_path = SKETCH_DIR / f"{stem}.png"

            if not pred_path.exists():
                print(f"[{method_name}] SKIP {stem}: {pred_path} 없음")
                continue
            if not gt_path.exists() or not matte_path.exists():
                print(f"[{method_name}] SKIP {stem}: GT/matte 없음")
                continue

            metrics = evaluate_single(
                str(pred_path), str(gt_path), str(matte_path),
                str(sketch_path) if sketch_path.exists() else None,
            )
            metrics["file"] = stem
            per_image.append(metrics)

            if inception is not None:
                pred_np  = load_image_np(str(pred_path))
                gt_np    = load_image_np(str(gt_path))
                matte_np = load_matte_np(str(matte_path), size=(pred_np.shape[1], pred_np.shape[0]))
                hair_p = _crop_hair(pred_np, matte_np)
                hair_g = _crop_hair(gt_np,   matte_np)
                if hair_p is not None and hair_g is not None:
                    feats_pred_hair.append(inception.extract(hair_p))
                    feats_gt_hair.append(inception.extract(hair_g))
                bnd_p = _crop_boundary(pred_np, matte_np)
                bnd_g = _crop_boundary(gt_np,   matte_np)
                if bnd_p is not None and bnd_g is not None:
                    feats_pred_bnd.append(inception.extract(bnd_p))
                    feats_gt_bnd.append(inception.extract(bnd_g))

        if not per_image:
            print(f"[{method_name}] 평가 파일 없음, 건너뜀")
            continue

        numeric_keys = [k for k in per_image[0] if k != "file"]
        means = {}
        stds  = {}
        for k in numeric_keys:
            m, s = mean_std([float(row[k]) for row in per_image])
            means[k] = m
            stds[k]  = s

        fid_hair = compute_fid(feats_pred_hair, feats_gt_hair)
        fid_bnd  = compute_fid(feats_pred_bnd,  feats_gt_bnd)

        all_results[method_name] = {
            "n": len(per_image),
            "means": means,
            "stds": stds,
            "fid_hair": fid_hair,
            "fid_boundary": fid_bnd,
            "per_image": per_image,
        }

    # ── 비교 출력 ──
    metric_keys = ["psnr_bg", "ssim_bg", "ssim_hair", "lpips_hair",
                   "boundary_ssim", "boundary_lpips", "edge_iou", "chamfer_dist",
                   "arcface_cos"]

    header = f"{'Metric':<20}" + "".join(f"{m:>22}" for m in all_results)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for k in metric_keys:
        row = f"{k:<20}"
        for method_name, res in all_results.items():
            v = res["means"].get(k, float("nan"))
            s = res["stds"].get(k, float("nan"))
            if np.isnan(v):
                row += f"{'nan':>22}"
            else:
                row += f"{v:>12.4f} ±{s:>7.4f}"
        print(row)

    print("-" * len(header))
    for label, key in [("FID (hair)", "fid_hair"), ("FID (boundary)", "fid_boundary")]:
        row = f"{label:<20}"
        for res in all_results.values():
            v = res[key]
            row += f"{v:>22.4f}" if not np.isnan(v) else f"{'nan':>22}"
        print(row)
    print("=" * len(header))

    # ── 저장 ──
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
