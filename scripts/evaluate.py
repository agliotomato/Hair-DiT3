"""
정량 평가 스크립트 — Hair-DiT2 실험 계획서 기준 전체 지표

Per-image metrics:
  배경 보존 (1-matte 영역):
    - PSNR_bg, SSIM_bg
  헤어 생성 품질 (matte 영역):
    - LPIPS_hair    : VGG 기반 지각 손실 (vs GT)
    - SSIM_hair     : 헤어 영역 구조 유사도 (vs GT)
  구조 충실도 (Sketch Fidelity):
    - Edge_IoU      : Canny(generated) ∩ Canny(sketch) / union, hair region 내
    - Chamfer_dist  : edge ↔ sketch edge 양방향 최소 거리 평균 (↓ 좋음)
  경계면:
    - Boundary_SSIM : dilate XOR erode 영역 SSIM
    - Boundary_LPIPS: boundary strip LPIPS (vs GT)
  얼굴 identity:
    - ArcFace_cos   : ArcFace cosine similarity (insightface 필요, 없으면 nan)

Dataset-level metrics (루프 후 계산):
  - FID            : InceptionV3 pool 특징 → Fréchet distance (hair crop)
  - Boundary_FID   : matte 0.1~0.9 strip crop FID

사용법:
  python scripts/evaluate.py \\
    --pred_dir results/batch_infer_final2 \\
    --gt_dir   dataset3/braid/img/test \\
    --matte_dir dataset3/braid/matte/test \\
    --sketch_dir dataset3/braid/sketch/test \\
    --method_name ours \\
    --output eval_results_ours.csv
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple

# ── 선택적 의존성 ──────────────────────────────────────────────────────────────
try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import lpips as lpips_lib
    _lpips_fn = lpips_lib.LPIPS(net="vgg")
    _lpips_fn.eval()
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    _lpips_fn = None

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from scipy.linalg import sqrtm as scipy_sqrtm
    HAS_SCIPY_LINALG = True
except ImportError:
    HAS_SCIPY_LINALG = False

try:
    import insightface
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_image_np(path: str) -> np.ndarray:
    """PIL → numpy [H, W, 3] uint8"""
    return np.array(Image.open(path).convert("RGB"))


def load_matte_np(path: str, size=None) -> np.ndarray:
    """PIL → numpy [H, W] float32 in [0, 1]"""
    m = Image.open(path).convert("L")
    if size:
        m = m.resize(size, Image.LANCZOS)
    return np.array(m).astype(np.float32) / 255.0


# ── Per-image metrics ─────────────────────────────────────────────────────────

def compute_psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """배경 영역(1-mask)에서 PSNR"""
    bg = (1 - mask)[..., None]
    p = (pred * bg).astype(np.float32)
    g = (gt   * bg).astype(np.float32)
    if not HAS_SKIMAGE:
        mse = np.mean((p - g) ** 2)
        return float(10 * np.log10(255 ** 2 / (mse + 1e-8)))
    return float(sk_psnr(g, p, data_range=255))


def compute_ssim_bg(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """배경 영역(1-mask)에서 SSIM"""
    if not HAS_SKIMAGE:
        return float("nan")
    bg = (1 - mask)[..., None]
    p = (pred * bg).astype(np.float32)
    g = (gt   * bg).astype(np.float32)
    return float(sk_ssim(g, p, data_range=255, channel_axis=-1))


def compute_ssim_hair(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """헤어 영역(mask)에서 SSIM — per-pixel map 평균"""
    if not HAS_SKIMAGE:
        return float("nan")
    p = (pred * mask[..., None]).astype(np.float32)
    g = (gt   * mask[..., None]).astype(np.float32)
    _, ssim_map = sk_ssim(g, p, data_range=255, channel_axis=-1, full=True)
    # mask 영역에서만 평균
    hair_pixels = mask > 0.1
    if hair_pixels.sum() == 0:
        return float("nan")
    return float(ssim_map[hair_pixels].mean())


def compute_lpips(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """헤어 영역(mask)에서 LPIPS (vs GT)"""
    if not HAS_LPIPS:
        return float("nan")
    def to_t(img, m):
        t = torch.from_numpy(img).float() / 127.5 - 1.0
        t = t.permute(2, 0, 1).unsqueeze(0)
        m_t = torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
        return t * m_t
    with torch.no_grad():
        val = _lpips_fn(to_t(pred, mask), to_t(gt, mask))
    return float(val.mean())


def compute_sketch_edge_iou(pred: np.ndarray, sketch: np.ndarray,
                             mask: np.ndarray) -> float:
    """생성 hair edge vs sketch IoU (hair region 내)"""
    if not HAS_CV2:
        return float("nan")
    m = (mask > 0.1).astype(np.uint8)
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
    pred_hair = pred_gray * m
    edge_pred = cv2.Canny(pred_hair, 50, 150)
    sk_gray = sketch if sketch.ndim == 2 else cv2.cvtColor(sketch, cv2.COLOR_RGB2GRAY)
    _, sk_bin = cv2.threshold(sk_gray, 128, 255, cv2.THRESH_BINARY)
    inter = np.logical_and(edge_pred > 0, sk_bin > 0).sum()
    union = np.logical_or( edge_pred > 0, sk_bin > 0).sum()
    return float(inter / (union + 1e-8))


def compute_chamfer_distance(pred: np.ndarray, sketch: np.ndarray,
                              mask: np.ndarray) -> float:
    """
    Chamfer Distance: Canny(generated) ↔ sketch edge 양방향 최소 거리 평균.
    단위: pixel. 낮을수록 sketch에 충실.
    """
    if not HAS_CV2 or not HAS_SCIPY:
        return float("nan")
    m = (mask > 0.1).astype(np.uint8)
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
    edge_pred = cv2.Canny(pred_gray * m, 50, 150)

    sk_gray = sketch if sketch.ndim == 2 else cv2.cvtColor(sketch, cv2.COLOR_RGB2GRAY)
    _, sk_bin = cv2.threshold(sk_gray, 128, 255, cv2.THRESH_BINARY)

    pred_pts = np.column_stack(np.where(edge_pred > 0))  # (N, 2)
    sk_pts   = np.column_stack(np.where(sk_bin   > 0))   # (M, 2)

    if len(pred_pts) == 0 or len(sk_pts) == 0:
        return float("nan")

    tree_sk   = cKDTree(sk_pts)
    tree_pred = cKDTree(pred_pts)
    d_p2s, _ = tree_sk.query(pred_pts)
    d_s2p, _ = tree_pred.query(sk_pts)
    return float((d_p2s.mean() + d_s2p.mean()) / 2.0)


def _get_boundary_mask(matte_np: np.ndarray) -> np.ndarray:
    """matte 0.1~0.9 사이의 경계 strip mask (float32)"""
    if not HAS_CV2:
        return np.zeros_like(matte_np)
    matte_u8 = (matte_np * 255).astype(np.uint8)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated  = cv2.dilate(matte_u8, kernel)
    eroded   = cv2.erode(matte_u8,  kernel)
    return ((dilated > 0) & (eroded == 0)).astype(np.float32)


def compute_boundary_ssim(pred: np.ndarray, gt: np.ndarray,
                           matte_np: np.ndarray) -> float:
    """경계면 영역 SSIM"""
    if not HAS_CV2 or not HAS_SKIMAGE:
        return float("nan")
    boundary = _get_boundary_mask(matte_np)
    p = (pred * boundary[..., None]).astype(np.float32)
    g = (gt   * boundary[..., None]).astype(np.float32)
    if boundary.sum() < 10:
        return float("nan")
    return float(sk_ssim(g, p, data_range=255, channel_axis=-1))


def compute_boundary_lpips(pred: np.ndarray, gt: np.ndarray,
                            matte_np: np.ndarray) -> float:
    """경계면 영역 LPIPS (vs GT)"""
    if not HAS_LPIPS or not HAS_CV2:
        return float("nan")
    boundary = _get_boundary_mask(matte_np)
    if boundary.sum() < 10:
        return float("nan")
    def to_t(img, m):
        t = torch.from_numpy(img).float() / 127.5 - 1.0
        t = t.permute(2, 0, 1).unsqueeze(0)
        m_t = torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
        return t * m_t
    with torch.no_grad():
        val = _lpips_fn(to_t(pred, boundary), to_t(gt, boundary))
    return float(val.mean())


def compute_arcface_cosine(pred: np.ndarray, gt: np.ndarray,
                            matte_np: np.ndarray) -> float:
    """
    ArcFace cosine similarity: 비헤어(얼굴) 영역 identity 보존도.
    insightface 미설치 시 nan 반환.
    """
    if not HAS_INSIGHTFACE:
        return float("nan")
    try:
        app = insightface.app.FaceAnalysis(allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=-1)  # CPU

        face_mask = ((1 - matte_np) > 0.5).astype(np.uint8)[..., None]
        pred_face = (pred * face_mask).astype(np.uint8)
        gt_face   = (gt   * face_mask).astype(np.uint8)

        faces_pred = app.get(cv2.cvtColor(pred_face, cv2.COLOR_RGB2BGR))
        faces_gt   = app.get(cv2.cvtColor(gt_face,   cv2.COLOR_RGB2BGR))

        if not faces_pred or not faces_gt:
            return float("nan")

        e1 = faces_pred[0].embedding
        e2 = faces_gt[0].embedding
        cos = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
        return cos
    except Exception:
        return float("nan")


# ── FID (InceptionV3 기반, torchvision) ──────────────────────────────────────

class InceptionFeatureExtractor:
    """InceptionV3 pool3 (2048-dim) 특징 추출기."""

    def __init__(self, device: str = "cpu"):
        from torchvision.models import inception_v3, Inception_V3_Weights
        from torchvision import transforms

        self.device = torch.device(device)
        self._transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.eval()
        # pool3 출력 hook
        self._feats: List[np.ndarray] = []
        model.avgpool.register_forward_hook(
            lambda m, i, o: self._feats.append(
                o.squeeze(-1).squeeze(-1).detach().cpu().float().numpy()
            )
        )
        self.model = model.to(self.device)

    @torch.no_grad()
    def extract(self, img_np: np.ndarray) -> np.ndarray:
        """img_np: [H,W,3] uint8 → (2048,) float32"""
        pil = Image.fromarray(img_np)
        x = self._transform(pil).unsqueeze(0).to(self.device)
        self._feats.clear()
        self.model(x)
        return self._feats[-1][0]  # (2048,)


def _compute_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                               mu2: np.ndarray, sigma2: np.ndarray,
                               eps: float = 1e-6) -> float:
    """Fréchet distance (FID 공식)"""
    if not HAS_SCIPY_LINALG:
        return float("nan")
    diff = mu1 - mu2
    # 수치 안정성: 작은 diagonal noise 추가
    offset = np.eye(sigma1.shape[0]) * eps
    covmean, _ = scipy_sqrtm(sigma1.dot(sigma2) + offset, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return fid


def compute_fid(feats_pred: List[np.ndarray],
                feats_gt:   List[np.ndarray]) -> float:
    """수집된 Inception feature list → FID scalar"""
    if len(feats_pred) < 2 or len(feats_gt) < 2:
        return float("nan")
    p = np.stack(feats_pred)  # (N, 2048)
    g = np.stack(feats_gt)
    mu_p, mu_g = p.mean(0), g.mean(0)
    sig_p = np.cov(p, rowvar=False) if len(p) > 1 else np.zeros((p.shape[1],) * 2)
    sig_g = np.cov(g, rowvar=False) if len(g) > 1 else np.zeros((g.shape[1],) * 2)
    return _compute_frechet_distance(mu_p, sig_p, mu_g, sig_g)


def _crop_hair(img_np: np.ndarray, matte_np: np.ndarray,
               min_size: int = 32) -> Optional[np.ndarray]:
    """matte 영역 bounding box crop → [H',W',3] uint8"""
    ys, xs = np.where(matte_np > 0.1)
    if len(ys) == 0:
        return None
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    if (y2 - y1) < min_size or (x2 - x1) < min_size:
        return None
    masked = (img_np * matte_np[..., None]).astype(np.uint8)
    return masked[y1:y2, x1:x2]


def _crop_boundary(img_np: np.ndarray,
                   matte_np: np.ndarray) -> Optional[np.ndarray]:
    """boundary strip crop"""
    boundary = _get_boundary_mask(matte_np)
    if boundary.sum() < 100:
        return None
    masked = (img_np * boundary[..., None]).astype(np.uint8)
    ys, xs = np.where(boundary > 0)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    return masked[y1:y2, x1:x2]


# ── Per-image 통합 평가 ────────────────────────────────────────────────────────

def evaluate_single(
    pred_path:   str,
    gt_path:     str,
    matte_path:  str,
    sketch_path: Optional[str] = None,
) -> Dict[str, float]:
    pred   = load_image_np(pred_path)
    gt     = load_image_np(gt_path)
    matte  = load_matte_np(matte_path, size=(pred.shape[1], pred.shape[0]))
    sketch = load_image_np(sketch_path) if sketch_path else None

    results: Dict[str, float] = {
        "psnr_bg":        compute_psnr(pred, gt, matte),
        "ssim_bg":        compute_ssim_bg(pred, gt, matte),
        "ssim_hair":      compute_ssim_hair(pred, gt, matte),
        "lpips_hair":     compute_lpips(pred, gt, matte),
        "boundary_ssim":  compute_boundary_ssim(pred, gt, matte),
        "boundary_lpips": compute_boundary_lpips(pred, gt, matte),
        "arcface_cos":    compute_arcface_cosine(pred, gt, matte),
    }
    if sketch is not None:
        results["edge_iou"]     = compute_sketch_edge_iou(pred, sketch, matte)
        results["chamfer_dist"] = compute_chamfer_distance(pred, sketch, matte)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Hair-DiT2 정량 평가")
    parser.add_argument("--pred_dir",    type=str, required=True)
    parser.add_argument("--gt_dir",      type=str, required=True)
    parser.add_argument("--matte_dir",   type=str, required=True)
    parser.add_argument("--sketch_dir",  type=str, default=None)
    parser.add_argument("--method_name", type=str, default="ours",
                        help="CSV에 기록할 method 컬럼 값")
    parser.add_argument("--output",      type=str, default="eval_results.csv")
    parser.add_argument("--output_format", choices=["csv", "json"],
                        default="csv")
    parser.add_argument("--no_fid",      action="store_true",
                        help="FID 계산 생략 (빠른 디버그용)")
    parser.add_argument("--device",      type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    pred_dir  = Path(args.pred_dir)
    gt_dir    = Path(args.gt_dir)
    matte_dir = Path(args.matte_dir)

    # FID용 InceptionV3 초기화
    inception = None
    if not args.no_fid:
        try:
            print(f"InceptionV3 로드 ({args.device})...")
            inception = InceptionFeatureExtractor(device=args.device)
            print("InceptionV3 로드 완료.")
        except Exception as e:
            print(f"[경고] InceptionV3 로드 실패 → FID 생략: {e}")

    pred_files = sorted(pred_dir.glob("*.png")) + sorted(pred_dir.glob("*.jpg"))
    all_metrics: List[Dict] = []
    feats_pred_hair, feats_gt_hair = [], []
    feats_pred_bnd,  feats_gt_bnd  = [], []

    for pred_path in pred_files:
        stem = pred_path.stem
        gt_path    = next(gt_dir.glob(f"{stem}.*"),    None)
        matte_path = next(matte_dir.glob(f"{stem}.*"), None)
        sketch_path = None
        if args.sketch_dir:
            sketch_path = next(Path(args.sketch_dir).glob(f"{stem}.*"), None)

        if gt_path is None or matte_path is None:
            print(f"SKIP {stem}: GT 또는 matte 없음")
            continue

        metrics = evaluate_single(
            str(pred_path), str(gt_path), str(matte_path),
            str(sketch_path) if sketch_path else None,
        )
        metrics["file"]   = stem
        metrics["method"] = args.method_name
        all_metrics.append(metrics)

        print(f"{stem}: " + ", ".join(
            f"{k}={v:.4f}" for k, v in metrics.items()
            if k not in ("file", "method") and not np.isnan(float(v))
        ))

        # Inception feature 수집 (FID용)
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

    if not all_metrics:
        print("평가할 파일 없음")
        return

    # ── Dataset-level FID 계산 ──
    fid_hair = compute_fid(feats_pred_hair, feats_gt_hair)
    fid_bnd  = compute_fid(feats_pred_bnd,  feats_gt_bnd)
    print(f"\nFID (hair):     {fid_hair:.4f}  (n={len(feats_pred_hair)})")
    print(f"FID (boundary): {fid_bnd:.4f}  (n={len(feats_pred_bnd)})")

    # ── 평균 출력 ──
    numeric_keys = [k for k in all_metrics[0].keys() if k not in ("file", "method")]
    print(f"\n=== 평균 (method={args.method_name}, n={len(all_metrics)}) ===")
    mean_dict: Dict[str, float] = {}
    for k in numeric_keys:
        vals = [float(m[k]) for m in all_metrics if not np.isnan(float(m[k]))]
        if vals:
            mean_dict[k] = float(np.mean(vals))
            std = float(np.std(vals))
            print(f"  {k}: {mean_dict[k]:.4f} ± {std:.4f}  (n={len(vals)})")
    if not np.isnan(fid_hair):
        print(f"  fid_hair: {fid_hair:.4f}")
    if not np.isnan(fid_bnd):
        print(f"  fid_boundary: {fid_bnd:.4f}")

    # ── 저장 ──
    if args.output_format == "json":
        import json
        out = {
            "method": args.method_name,
            "n": len(all_metrics),
            "means": mean_dict,
            "fid_hair": fid_hair,
            "fid_boundary": fid_bnd,
            "per_image": all_metrics,
        }
        out_path = Path(args.output).with_suffix(".json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n결과 저장 (JSON): {out_path}")
    else:
        import csv
        fieldnames = ["method", "file"] + numeric_keys
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\n결과 저장 (CSV): {args.output}")

    print("\n[참고] ArcFace: insightface 미설치 시 nan. 설치: pip install insightface")
    print("[참고] FID: nan이면 scipy.linalg 또는 torchvision 확인 필요")


if __name__ == "__main__":
    main()
