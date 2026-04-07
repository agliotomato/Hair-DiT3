"""
정량 평가 스크립트

평가 지표:
  배경 보존 (1-matte 영역):
    - PSNR  : 목표 35dB 이상
    - SSIM  : 목표 0.95 이상

  헤어 생성 품질 (matte 영역):
    - LPIPS : 낮을수록 좋음 (VGG 기반)

  구조 정합도:
    - Sketch-Edge IoU : 생성 헤어 엣지 맵 vs 입력 스케치 IoU

  경계면 자연도:
    - Boundary SSIM : dilate(matte) XOR erode(matte) 영역 SSIM
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List

try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def load_image_np(path: str) -> np.ndarray:
    """PIL → numpy [H, W, 3] uint8"""
    return np.array(Image.open(path).convert("RGB"))


def load_matte_np(path: str, size=None) -> np.ndarray:
    """PIL → numpy [H, W] float32 in [0, 1]"""
    m = Image.open(path).convert("L")
    if size:
        m = m.resize(size, Image.LANCZOS)
    return np.array(m).astype(np.float32) / 255.0


def compute_psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """배경 영역(1-mask)에서 PSNR"""
    bg_mask = (1 - mask)[..., None]  # [H, W, 1]
    p = (pred * bg_mask).astype(np.float32)
    g = (gt   * bg_mask).astype(np.float32)
    if not HAS_SKIMAGE:
        mse = np.mean((p - g) ** 2)
        return float(10 * np.log10(255**2 / (mse + 1e-8)))
    return float(sk_psnr(g, p, data_range=255))


def compute_ssim(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """배경 영역(1-mask)에서 SSIM"""
    if not HAS_SKIMAGE:
        return float("nan")
    bg_mask = (1 - mask)[..., None]
    p = (pred * bg_mask).astype(np.float32)
    g = (gt   * bg_mask).astype(np.float32)
    return float(sk_ssim(g, p, multichannel=True, data_range=255,
                          channel_axis=-1))


def compute_lpips(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """헤어 영역(mask)에서 LPIPS"""
    if not HAS_LPIPS:
        return float("nan")
    loss_fn = lpips.LPIPS(net="vgg")
    def to_tensor(img, m):
        t = torch.from_numpy(img).float() / 127.5 - 1.0
        t = t.permute(2, 0, 1).unsqueeze(0) * torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
        return t
    with torch.no_grad():
        val = loss_fn(to_tensor(pred, mask), to_tensor(gt, mask))
    return float(val.mean())


def compute_sketch_edge_iou(pred: np.ndarray, sketch: np.ndarray, mask: np.ndarray) -> float:
    """생성 이미지 헤어 엣지 vs 입력 스케치 IoU"""
    if not HAS_CV2:
        return float("nan")
    # 헤어 영역 grayscale
    m = mask.astype(np.uint8)
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
    pred_hair = pred_gray * m
    edge_pred = cv2.Canny(pred_hair, 50, 150)

    # 스케치 binarize
    sk_gray = sketch if sketch.ndim == 2 else cv2.cvtColor(sketch, cv2.COLOR_RGB2GRAY)
    _, sk_bin = cv2.threshold(sk_gray, 128, 255, cv2.THRESH_BINARY)

    # IoU
    inter = np.logical_and(edge_pred > 0, sk_bin > 0).sum()
    union = np.logical_or( edge_pred > 0, sk_bin > 0).sum()
    return float(inter / (union + 1e-8))


def compute_boundary_ssim(pred: np.ndarray, gt: np.ndarray, matte_np: np.ndarray) -> float:
    """경계면 영역 SSIM"""
    if not HAS_CV2 or not HAS_SKIMAGE:
        return float("nan")
    matte_u8 = (matte_np * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(matte_u8, kernel)
    eroded  = cv2.erode(matte_u8, kernel)
    boundary = ((dilated > 0) & (eroded == 0)).astype(np.float32)
    p = (pred * boundary[..., None]).astype(np.float32)
    g = (gt   * boundary[..., None]).astype(np.float32)
    if boundary.sum() < 10:
        return float("nan")
    return float(sk_ssim(g, p, multichannel=True, data_range=255, channel_axis=-1))


def evaluate_single(
    pred_path: str,
    gt_path: str,
    matte_path: str,
    sketch_path: str = None,
) -> Dict[str, float]:
    pred   = load_image_np(pred_path)
    gt     = load_image_np(gt_path)
    matte  = load_matte_np(matte_path, size=(pred.shape[1], pred.shape[0]))
    sketch = load_image_np(sketch_path) if sketch_path else None

    results = {
        "psnr_bg":        compute_psnr(pred, gt, matte),
        "ssim_bg":        compute_ssim(pred, gt, matte),
        "lpips_hair":     compute_lpips(pred, gt, matte),
        "boundary_ssim":  compute_boundary_ssim(pred, gt, matte),
    }
    if sketch is not None:
        results["sketch_edge_iou"] = compute_sketch_edge_iou(pred, sketch, matte)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="EHC 모델 정량 평가")
    parser.add_argument("--pred_dir",   type=str, required=True, help="예측 이미지 디렉토리")
    parser.add_argument("--gt_dir",     type=str, required=True, help="GT 이미지 디렉토리")
    parser.add_argument("--matte_dir",  type=str, required=True, help="헤어 매트 디렉토리")
    parser.add_argument("--sketch_dir", type=str, default=None,  help="스케치 디렉토리 (optional)")
    parser.add_argument("--output",     type=str, default="eval_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    pred_dir  = Path(args.pred_dir)
    gt_dir    = Path(args.gt_dir)
    matte_dir = Path(args.matte_dir)

    pred_files = sorted(pred_dir.glob("*.png")) + sorted(pred_dir.glob("*.jpg"))
    all_metrics: List[Dict] = []

    for pred_path in pred_files:
        stem = pred_path.stem
        gt_path     = next(gt_dir.glob(f"{stem}.*"), None)
        matte_path  = next(matte_dir.glob(f"{stem}.*"), None)
        sketch_path = None
        if args.sketch_dir:
            sketch_path = next(Path(args.sketch_dir).glob(f"{stem}.*"), None)

        if gt_path is None or matte_path is None:
            print(f"SKIP {stem}: GT 또는 matte 없음")
            continue

        metrics = evaluate_single(
            str(pred_path), str(gt_path), str(matte_path),
            str(sketch_path) if sketch_path else None
        )
        metrics["file"] = stem
        all_metrics.append(metrics)

        print(f"{stem}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "file"))

    if not all_metrics:
        print("평가할 파일 없음")
        return

    # 평균 출력
    keys = [k for k in all_metrics[0].keys() if k != "file"]
    print("\n=== 평균 ===")
    for k in keys:
        vals = [m[k] for m in all_metrics if not np.isnan(m.get(k, float("nan")))]
        if vals:
            print(f"  {k}: {np.mean(vals):.4f} (n={len(vals)})")

    # CSV 저장
    import csv
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file"] + keys)
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()
