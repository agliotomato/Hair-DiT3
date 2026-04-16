"""
Hypothesis 2: Background Context Dependency 실험

동일한 피험자의 sketch + matte + 얼굴 픽셀을 고정하고,
순수 배경 영역(non-hair, non-face)만 3가지 조건으로 교체.
모델이 background context에 의존하는지 확인.

조건:
  white_bg   - 단색 (흰색) 배경
  texture_bg - 체커보드 텍스처 배경
  complex_bg - 인물 없는 복잡한 배경 씬

대조군 원칙:
  변수 = 순수 배경 픽셀 1가지.
  sketch, matte, 얼굴 픽셀, seed, compositor 파라미터 모두 고정.
  m_subj = clip(m_hair + m_face, 0, 1) 로 얼굴·헤어 영역을 함께 보존.
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import urllib.request
import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from src.models.hair_s2i_net import HairS2INet


REFERENCES = ["braid_2534", "braid_2562", "braid_2574", "braid_2653"]
COMPLEX_BG_IMAGE = "dataset3/backgrounds/indoor_scene.png"  # 인물 없는 배경 씬

SIZE = 512
SOFT_BLUR_RADIUS = 21  # Soft Mask 경계 blur 커널 크기 (홀수 강제)

_LANDMARKER_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_LANDMARKER_PATH = ".cache/face_landmarker.task"


def _ensure_landmarker():
    """MediaPipe Face Landmarker 모델 파일이 없으면 자동 다운로드."""
    from pathlib import Path
    p = Path(_LANDMARKER_PATH)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Face Landmarker 모델 다운로드 중: {_LANDMARKER_URL}")
        urllib.request.urlretrieve(_LANDMARKER_URL, p)
        print(f"  완료: {p}")


# ---------------------------------------------------------------------------
# Subject mask: m_subj = clip(m_hair + m_face, 0, 1)  — Soft Mask
# ---------------------------------------------------------------------------

def make_face_mask_soft(face_pil: Image.Image, size: int = 512,
                        blur_radius: int = SOFT_BLUR_RADIUS,
                        dilate_px: int = 15,
                        matte_arr: np.ndarray = None,
                        debug_path: str = None) -> tuple:
    """
    얼굴 영역 Soft Mask 생성. 2단계 시도 후 미감지 시 zero mask 반환.
      1) MediaPipe Face Landmarker (Tasks API, >= 0.10) — 478 랜드마크 Convex Hull
      2) MediaPipe FaceMesh (solutions API, < 0.10)   — 478 랜드마크 Convex Hull
      감지 실패 (뒷모습/옆모습) → zero mask 반환 (m_subj = m_hair only)
    → dilation으로 마스크 여백 확보 후 Gaussian blur로 경계 부드럽게 처리.

    Args:
        dilate_px:   Convex Hull 마스크를 팽창시킬 픽셀 수 (턱선·이마 여백)
        debug_path:  지정 시 마스크 오버레이 이미지를 해당 경로에 저장
    반환: ([H, W] float32 0~1, face_detected: bool)
    """
    face_arr = np.array(face_pil.convert("RGB").resize((size, size)))
    mask = np.zeros((size, size), dtype=np.float32)
    detected = False
    method = "no_face"

    # 1차: MediaPipe Tasks API (mediapipe >= 0.10) — 478 landmarks
    try:
        _ensure_landmarker()
        from mediapipe.tasks import python as _mp_python
        from mediapipe.tasks.python import vision as _mp_vision

        base_options = _mp_python.BaseOptions(model_asset_path=_LANDMARKER_PATH)
        options = _mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        with _mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_arr)
            result = landmarker.detect(mp_img)
        print(f"  [face_mask] Tasks API: {len(result.face_landmarks)} face(s) detected")
        if result.face_landmarks:
            lms = result.face_landmarks[0]
            pts = np.array(
                [[int(lm.x * size), int(lm.y * size)] for lm in lms], dtype=np.int32
            )
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 1.0)
            detected = True
            method = "mediapipe_tasks"
    except Exception as e:
        print(f"  [face_mask] Tasks API 실패: {e}")

    # 2차: MediaPipe FaceMesh solutions API (mediapipe < 0.10)
    if not detected:
        try:
            with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True
            ) as face_mesh:
                results = face_mesh.process(face_arr)
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                pts = np.array(
                    [[int(lm.x * size), int(lm.y * size)] for lm in lms], dtype=np.int32
                )
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(mask, hull, 1.0)
                detected = True
                method = "mediapipe_solutions"
        except Exception as e:
            print(f"  [face_mask] FaceMesh 실패: {e}")

    # 감지 실패: 뒷모습/옆모습으로 판단 → zero mask (m_subj = m_hair only)
    if not detected:
        print(f"  [face_mask] method={method} → m_subj = m_hair only (뒷모습/옆모습 추정)")
        return np.zeros((size, size), dtype=np.float32), False

    print(f"  [face_mask] method={method}")

    # Dilation: Convex Hull이 턱선·이마를 빠뜨리는 경우 여백 확보
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel)

    # Soft mask: Gaussian blur로 경계 점진 처리
    k = blur_radius | 1
    blurred = cv2.GaussianBlur(mask, (k, k), 0)
    if blurred.max() > 0:
        blurred = blurred / blurred.max()
    result = np.clip(blurred, 0.0, 1.0)

    # 디버그: 마스크 오버레이 저장
    if debug_path:
        overlay = face_arr.copy().astype(np.float32)
        overlay[..., 1] = np.clip(overlay[..., 1] + result * 120, 0, 255)  # 초록 채널 강조
        debug_img = Image.fromarray(overlay.astype(np.uint8))
        Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
        debug_img.save(debug_path)

    return result, True


def make_subject_mask(matte_pil: Image.Image, face_pil: Image.Image,
                      size: int = 512) -> np.ndarray:
    """
    m_subj = clip(m_hair + m_face_soft, 0, 1)
    m_hair : MatteCNN 출력 (이미 soft)
    m_face : MediaPipe FaceMesh + Gaussian blur (soft)
            얼굴 미감지(뒷모습/옆모습) → m_face = 0, m_subj = m_hair only
    반환: [H, W] float32, 0~1  — Alpha Blending용 Soft Mask
    """
    m_hair = np.array(matte_pil.convert("L").resize((size, size)), dtype=np.float32) / 255.0
    m_face, face_detected = make_face_mask_soft(face_pil, size, matte_arr=m_hair)
    if not face_detected:
        return m_hair
    return np.clip(m_hair + m_face, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Background 생성 함수 — 모두 m_subj 기반
# ---------------------------------------------------------------------------

def make_white_bg(face_pil: Image.Image, matte_pil: Image.Image) -> Image.Image:
    """순수 배경 영역 → 흰색 / 얼굴·헤어 영역 → 원본 유지."""
    face = np.array(face_pil.convert("RGB").resize((SIZE, SIZE)), dtype=np.float32)
    m    = make_subject_mask(matte_pil, face_pil)[..., None]
    white = np.ones((SIZE, SIZE, 3), dtype=np.float32) * 255.0
    result = face * m + white * (1.0 - m)
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


def make_texture_bg(face_pil: Image.Image, matte_pil: Image.Image, tile: int = 8) -> Image.Image:
    """순수 배경 영역 → 체커보드 / 얼굴·헤어 영역 → 원본 유지."""
    face = np.array(face_pil.convert("RGB").resize((SIZE, SIZE)), dtype=np.float32)
    m    = make_subject_mask(matte_pil, face_pil)[..., None]
    idx  = np.indices((SIZE, SIZE))
    gray = np.where((idx[0] // tile + idx[1] // tile) % 2, 180, 100).astype(np.float32)
    tex  = np.stack([gray, gray, gray], axis=-1)
    result = face * m + tex * (1.0 - m)
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


def make_complex_bg(face_pil: Image.Image, matte_pil: Image.Image, bg_scene_pil: Image.Image) -> Image.Image:
    """순수 배경 영역 → 복잡한 배경 씬 / 얼굴·헤어 영역 → 원본 유지."""
    face  = np.array(face_pil.convert("RGB").resize((SIZE, SIZE)), dtype=np.float32)
    m     = make_subject_mask(matte_pil, face_pil)[..., None]
    scene = np.array(bg_scene_pil.convert("RGB").resize((SIZE, SIZE)), dtype=np.float32)
    result = face * m + scene * (1.0 - m)
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def make_comparison_grid(rows: list[tuple[str, Image.Image, Image.Image]]) -> Image.Image:
    """
    rows: [(label, input_bg, result), ...]
    각 행: [label | input_bg | result] 3열 구성
    """
    label_w = 120
    cell_w, cell_h = SIZE, SIZE
    n_rows = len(rows)
    total_w = label_w + cell_w * 2 + 6
    total_h = cell_h * n_rows + 2 * n_rows

    grid = Image.new("RGB", (total_w, total_h), (30, 30, 30))

    from PIL import ImageDraw
    draw = ImageDraw.Draw(grid)

    for i, (label, bg_img, out_img) in enumerate(rows):
        y = i * (cell_h + 2)
        draw.text((4, y + cell_h // 2 - 6), label, fill=(220, 220, 220))
        grid.paste(bg_img.resize((cell_w, cell_h)), (label_w, y))
        grid.paste(out_img.resize((cell_w, cell_h)), (label_w + cell_w + 3, y))

    return grid


# ---------------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Hypothesis 2: Background Context Dependency 실험")
    parser.add_argument("--pretrained_model", type=str,
                        default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--checkpoint",       type=str, required=True)
    parser.add_argument("--data_root",        type=str, default="dataset3/braid")
    parser.add_argument("--complex_bg",       type=str, default=COMPLEX_BG_IMAGE,
                        help="complex_bg 조건에 쓸 인물 없는 배경 이미지 경로")
    parser.add_argument("--output_dir",       type=str, default="results/hypothesis2")
    parser.add_argument("--num_steps",        type=int,   default=28)
    parser.add_argument("--guidance",         type=float, default=7.5)
    parser.add_argument("--controlnet_scale", type=float, default=1.0)
    parser.add_argument("--seed",             type=int,   default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.pretrained_model, args.checkpoint, device)

    bg_scene = Image.open(args.complex_bg).convert("RGB")

    for ref_id in REFERENCES:
        print(f"\n======== {ref_id} ========")
        ref_dir = output_dir / ref_id
        ref_dir.mkdir(parents=True, exist_ok=True)

        face   = Image.open(data_root / "img"    / "test" / f"{ref_id}.png").convert("RGB")
        sketch = Image.open(data_root / "sketch" / "test" / f"{ref_id}.png").convert("RGB")
        matte  = Image.open(data_root / "matte"  / "test" / f"{ref_id}.png").convert("L")

        # 첫 피험자만 face mask 디버그 이미지 저장
        debug_path = str(ref_dir / "debug_face_mask.png") if ref_id == REFERENCES[0] else None
        if debug_path:
            _, face_det = make_face_mask_soft(face, debug_path=debug_path)
            print(f"  [debug] face mask 저장: {debug_path} (detected={face_det})")

        backgrounds = {
            "white_bg":   make_white_bg(face, matte),
            "texture_bg": make_texture_bg(face, matte),
            "complex_bg": make_complex_bg(face, matte, bg_scene),
        }

        grid_rows = []

        for name, bg in backgrounds.items():
            bg_save_path = ref_dir / f"{name}_input_bg.png"
            bg.save(bg_save_path)

            out_path = ref_dir / f"{name}.png"
            if out_path.exists():
                print(f"[skip] {out_path}")
                result = Image.open(out_path).convert("RGB")
            else:
                print(f"[{ref_id}] {name} 추론 ...")
                result = model.inference(
                    background=bg,
                    sketch=sketch,
                    matte=matte,
                    num_steps=args.num_steps,
                    guidance_scale=args.guidance,
                    controlnet_scale=args.controlnet_scale,
                    size=(SIZE, SIZE),
                    seed=args.seed,
                )
                result.save(out_path)
                print(f"  저장: {out_path}")

            grid_rows.append((name, bg, result))

        grid = make_comparison_grid(grid_rows)
        grid_path = ref_dir / "comparison_grid.png"
        grid.save(grid_path)
        print(f"  그리드 저장: {grid_path}")

    print("\n완료.")


if __name__ == "__main__":
    main()
